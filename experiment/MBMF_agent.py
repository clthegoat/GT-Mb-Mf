import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from .models.model_based import trans_model, reward_model, value_model, actor_model, critic_model
from .MPC_agent import MPC_agent
from .MVE_agent import MVE_agent




class Memory():

    data_pointer = 0
    isfull = False

    def __init__(self, capacity):
        self.memory = np.empty(capacity, dtype=object)
        self.capacity = capacity
        self.count = 0

    def update(self, transition):
        self.memory[self.data_pointer] = transition
        self.data_pointer += 1
        self.count += 1
        if self.data_pointer == self.capacity:
            self.data_pointer = 0
            self.isfull = True

    def all_sample(self, batch_size):
        return np.random.choice(self.memory[0:self.count], batch_size)

    def MB_sample(self, batch_size, trail_len, K):
        memory = self.memory.tolist()
        MB_memory = [tran for tran in memory if tran.t < trail_len - K] # not sure whether < or <=
        MB_memory = np.array(MB_memory)
        return np.random.choice(MB_memory[0:self.count], batch_size)

    def MF_sample(self, batch_size, trail_len, K):
        memory = self.memory.tolist()
        MF_memory = [tran for tran in memory if tran.t >= trail_len - K]  # not sure whether >= or >
        MF_memory = np.array(MF_memory)
        return np.random.choice(MF_memory[0:self.count], batch_size)


class MBMF_agent(MVE_agent):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        self.gamma = self.conf.train.gamma

        self.memory = Memory(self.conf.data.mem_capacity)

        # agent
        self.mb_agent = MPC_agent(self.conf)
        self.mf_agent = MVE_agent(self.conf)

        # here we temporarily fix K
        self.K = self.conf.train.K

    def select_action(self, num_step, state, mode, exploration):
        '''
        according to yunkao, there are two choices for selecting action
        1) select actions w.r.t step
        2) always use MF to select action
        :return:
        '''

        # first choice
        if num_step <= self.trail_len - self.K:
            return self.mb_agent.select_action(num_step, state, mode, exploration)
        else:
            return self.mf_agent.select_action(state)
        # # second choice
        # return self.mf_agent.select_action(state)

    def store_transition(self, transition):
        return self.memory.update(transition)

    def update(self):

        self.training_step += 1

        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'

        self.trans_model.to(device)
        self.reward_model.to(device)

        # update transition and reward model, data sampled from all memory
        # data preparation
        all_transitions = self.memory.all_sample(self.conf.data.mem_batchsize)
        all_s_a = torch.tensor([t.s_a for t in all_transitions], dtype=torch.float).view(-1, self.dim_state_action)
        all_s_ = torch.tensor([t.s_ for t in all_transitions], dtype=torch.float).view(-1, self.dim_state)
        # actually train cost
        all_r = -torch.tensor([t.r for t in all_transitions], dtype=torch.float).view(-1, 1)
        all_s_a, all_s_, all_r = all_s_a.to(device), all_s_.to(device), all_r.to(device)

        # update transition model
        pred_state = self.trans_model(all_s_a)
        trans_loss = F.mse_loss(pred_state, all_s_)
        print("transition loss: {}".format(trans_loss.item()))

        self.optimizer_t.zero_grad()
        trans_loss.backward()
        self.optimizer_t.step()

        # update reward model
        pred_reward = self.reward_model(all_s_a)
        reward_loss = F.mse_loss(pred_reward, all_r)
        print("reward loss: {}".format(reward_loss.item()))

        self.optimizer_r.zero_grad()
        reward_loss.backward()
        self.optimizer_r.step()


        # update model-based models
        mb_transitions = self.memory.MB_sample(self.batch_size, self.trail_len, self.K)
        mb_s = torch.tensor([t.s for t in mb_transitions], dtype=torch.float).view(-1, self.dim_state)
        mb_s_a = torch.tensor([t.s_a for t in mb_transitions], dtype=torch.float).view(-1, self.dim_state_action)
        mb_s_ = torch.tensor([t.s_ for t in mb_transitions], dtype=torch.float).view(-1, self.dim_state)
        # actually train cost
        mb_r = -torch.tensor([t.r for t in mb_transitions], dtype=torch.float).view(-1, 1)
        mb_t = [t.t for t in mb_transitions]
        mb_s, mb_s_a, mb_s_, mb_r = mb_s.to(device), mb_s_a.to(device), mb_s_.to(device), mb_r.to(device)

        # get q-value target
        with torch.no_grad():
            q_target = self.critic_model(mb_s_a)

        q_pred = torch.zeros(self.batch_size)

        for i in range(self.batch_size):
            time_step = mb_t[i]
            cur_state = mb_s[i]
            j = 0
            while time_step < self.trail_len - self.K:  # here not sure whether < or <=
                action = self.select_action(time_step, cur_state, mode=2, exploration=0).to(device)
                state_action = torch.cat((cur_state, action))
                reward = self.reward_model(state_action)
                cur_state = self.trans_model(state_action)
                q_pred[i] += reward * self.gamma**j
                time_step += 1
                j += 1
            action_H = self.select_action(time_step, cur_state, mode=2, exploration=0).to(device)
            state_action_H = torch.cat((cur_state, action_H))
            q_pred[i] += self.critic_model(state_action_H) * self.gamma ** j

        # update mf-critic model
        mb_critic_loss = F.mse_loss(q_target, q_pred)
        self.optimizer_c.zero_grad()
        mb_critic_loss.backward()
        self.optimizer_c.step()


        #  update model-free models
        mf_transitions = self.memory.MF_sample(self.batch_size, self.trail_len, self.K)
        mf_s = torch.tensor([t.s for t in mf_transitions], dtype=torch.float).view(-1, self.dim_state)
        mf_s_a = torch.tensor([t.s_a for t in mf_transitions], dtype=torch.float).view(-1, self.dim_state_action)
        mf_s_ = torch.tensor([t.s_ for t in mf_transitions], dtype=torch.float).view(-1, self.dim_state)
        # actually train cost
        mf_r = -torch.tensor([t.r for t in mf_transitions], dtype=torch.float).view(-1, 1)
        mf_t = [t.t for t in mf_transitions]
        mf_s, mf_s_a, mf_s_, mf_r = mf_s.to(device), mf_s_a.to(device), mf_s_.to(device), mf_r.to(device)

        actor_loss = self.actor_learn(mf_s)
        critic_loss = self.critic_learn(mf_s_a, mf_r, mf_s_)










