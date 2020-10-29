from numpy.core.defchararray import index
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from models.model_based import trans_model, reward_model, value_model, actor_model, critic_model
from MPC_agent import MPC_agent
from MVE_agent import MVE_agent
from collections import namedtuple



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

    # def MB_sample(self, batch_size, trail_len, K):
    #     memory = self.memory.tolist()
    #     print(trail_len)
    #     print(K)
    #     # print(memory)
    #     MB_memory = [tran for tran in memory if (tran is not None) and (tran.t < (trail_len - K))] # not sure whether < or <=
    #     # MB_memory = np.array(MB_memory,dtype=tuple)
    #     # print(MB_memory)
    #     number_of_rows = len(MB_memory)
    #     print(number_of_rows)
    #     random_indices = np.random.choice(number_of_rows, size=batch_size)
    #     random_MB_sample = []
    #     for i in random_indices:
    #         random_MB_sample.append(MB_memory[i])
    #     print(random_MB_sample)
    #     # return np.random.choice(MB_memory[0:self.count], batch_size)
    #     return random_MB_sample

    def MB_sample(self, batch_size, trail_len, K):
        memory = self.memory.tolist()
        print(type(memory))
        print(trail_len)
        print(K)
        # print(memory)
        mb_transition = namedtuple('mb_transition', ['s', 'a', 's_a', 's_', 'r', 't', 'done'])
        # MB_memory = np.array([tran for tran in memory if (tran is not None) and (tran.t < (trail_len - K))], dtype=object) # not sure whether < or <=
        i = 0
        MB_memory = np.empty(1000, dtype=object)
        # MB_memory = []
        for tran in memory:
            if (tran is not None) and (tran.t < (trail_len - K)):
                MB_memory[i] = mb_transition(tran.s, tran.a, tran.s_a, tran.s_, tran.r, tran.t, tran.done)
                # MB_memory = np.append(MB_memory, mb_transition(tran.s, tran.a, tran.s_a, tran.s_, tran.r, tran.t, tran.done))
                i = i + 1
        new_MB_memory = MB_memory[np.s_[:i:1]]
        # i = 0
        # index = []
        # for tran in MB_memory:
        #     if tran == None:
        #         index.append(i)
        #     i = i + 1
        # new_MB_memory2 = np.delete(MB_memory, index)
        # print(new_MB_memory==new_MB_memory2)
        # print("yes")
        return np.random.choice(new_MB_memory[0:self.count], batch_size)
        # return random_MB_sample

    def MF_sample(self, batch_size, trail_len, K):
        memory = self.memory.tolist()
        MF_memory = [tran for tran in memory if tran.t >= trail_len - K]  # not sure whether >= or >
        MF_memory = np.array(MF_memory)
        return np.random.choice(MF_memory[0:self.count], batch_size)

    def judge_sample(self, trail_len, K):
        memory = self.memory.tolist()
        trans_memory = [tran for tran in memory if tran.t == trail_len - K]
        trans_memory = np.array(trans_memory)
        return trans_memory


class MBMF_agent(MVE_agent):

    def __init__(self, conf):
        super().__init__(conf)
        # self.conf = conf

        # self.gamma = self.conf.train.gamma

        self.memory = Memory(self.conf.data.mem_capacity)

        # agent
        self.mb_agent = MPC_agent(self.conf)
        self.mf_agent = MVE_agent(self.conf)

        # here we temporarily fix K
        self.K = self.conf.train.K
        self.c1 = self.conf.train.c1
        self.c2 = self.conf.train.c2
        # self.trail_len = self.conf.trail_len # steps in each trail
        # self.batch_size = self.conf.data.mem_batchsize

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

    def sample_transitions(self, sampleType):
        if sampleType == "all":
            transitions = self.memory.all_sample(batch_size=self.batch_size)
        elif sampleType == "MB":
            transitions = self.memory.MB_sample(batch_size=self.batch_size, trail_len=self.trail_len, K=self.K)
        elif sampleType == "MF":
            transitions = self.memory.MF_sample(batch_size=self.batch_size, trail_len=self.trail_len, K=self.K)
        elif sampleType == "judge":
            transitions = self.memory.judge_sample(batch_size=self.batch_size, trail_len=self.trail_len, K=self.K)
        else:
            raise ValueError
        s = torch.from_numpy(np.vstack((t.s for t in transitions if t is not None))).float().to(self.device)
        # s = torch.tensor([t.s for t in transitions], dtype=torch.float).view(-1, self.dim_state).to(self.device)
        a = torch.tensor([t.a for t in transitions], dtype=torch.float).view(-1, self.dim_action).to(self.device)
        s_a = torch.tensor([t.s_a for t in transitions], dtype=torch.float).view(-1, self.dim_state_action).to(self.device)
        s_ = torch.tensor([t.s_ for t in transitions], dtype=torch.float).view(-1, self.dim_state).to(self.device)
        r = torch.tensor([t.r for t in transitions], dtype=torch.float).view(-1, 1).to(self.device)
        t = torch.tensor([t.t for t in transitions], dtype=torch.float).view(-1, 1).to(self.device)
        return s, a, s_a, s_, r, t, #dones

    def update(self):

        self.training_step += 1

        """ update transition and reward model, data sampled from all memory"""
        _, _, all_s_a, all_s_, all_r, _ = self.sample_transitions("all")
        
        # update transition model
        trans_loss = self.trans_learn(all_s_a, all_s_)
        print("transition loss: {}".format(trans_loss))
        # update reward model
        reward_loss = self.reward_learn(all_s_a, all_r)
        print("reward loss: {}".format(reward_loss))

        """ update critic and actor model, data sampled from MB memory"""
        mb_s, _, mb_s_a, _, _, mb_t = self.sample_transitions("MB")
        mb_critic_loss, mb_actor_loss = self.MB_learn(mb_s, mb_s_a, mb_t)
        print("MB actor loss: {}".format(mb_actor_loss))
        print("MB critic loss: {}".format(mb_critic_loss))

        """ update critic and actor model, data sampled from MF memory"""
        mf_s, _, mf_s_a, mf_s_, mf_r, _ = self.sample_transitions("MF")
        mf_critic_loss, mf_actor_loss = self.MF_learn(mf_s,mf_s_a,mf_s_,mf_r)
        print("MF actor loss: {}".format(mf_actor_loss))
        print("MF critic loss: {}".format(mf_critic_loss))

        """ automatic transformation"""
        tk_s, _, tk_s_a, tk_s_, tk_r, _ = self.sample_transitions("judge")
        self.Auto_Transform(tk_s,tk_s_a,tk_s_,tk_r)
        
    
    def MB_learn(self,states,states_actions,time_steps):
        # q prediction and target
        q_pred = self.critic_local(states_actions)
        q_target = torch.zeros(self.batch_size)

        # policy prediction and target
        a_target = torch.zeros(self.batch_size)
        a_pred = torch.zeros(self.batch_size)

        for i in range(self.batch_size):
            time_step = time_steps[i]
            cur_state = states[i]
            j = 0
            while time_step < self.trail_len - self.K:  # here not sure whether < or <=
                action = torch.tensor(self.select_action(time_step, cur_state, mode=2, exploration=0), dtype=torch.float)
                state_action = torch.cat((cur_state, action))
                # record action (only record target action and predicted action on first step)
                if j==0:
                    a_target[i] = action
                    a_pred[i] = self.actor_local(cur_state)

                reward = self.reward_model(state_action)
                cur_state = self.trans_model(state_action)
                q_target[i:i+1] += reward * self.gamma**j
                time_step += 1
                j += 1
            action_H = torch.tensor(self.select_action(time_step, cur_state, mode=2, exploration=0), dtype=torch.float)
            state_action_H = torch.cat((cur_state, action_H))

            with torch.no_grad():
                q_target[i:i+1] += self.critic_target(state_action_H) * self.gamma ** j

        # update mb-critic model
        q_target, q_pred = q_target.view([-1, 1]), q_pred.view([-1, 1])
        mb_critic_loss = F.mse_loss(q_target, q_pred)
        self.optimizer_c.zero_grad()
        mb_critic_loss.backward()
        self.optimizer_c.step()
        self.soft_update_of_target_network(self.critic_local, self.critic_target, self.tau)

        # update mb-actor model
        a_target, a_pred = a_target.view([-1, 1]), a_pred.view([-1, 1])
        mb_actor_loss = F.mse_loss(a_pred, a_target)
        self.optimizer_a.zero_grad()
        mb_actor_loss.backward()
        self.optimizer_a.step()
        self.soft_update_of_target_network(self.actor_local, self.actor_target, self.tau)

        return mb_critic_loss, mb_actor_loss
   
    def MF_learn(self,states,states_actions,next_states,rewards):
        # update mf-actor model
        mf_actor_loss = self.actor_learn(states)
        # update mf-critic model
        actions_pred = self.actor_target(next_states)
        q_target = rewards + self.gamma*self.critic_target(torch.cat((next_states, actions_pred), 1))
        q_pred = self.critic_local(states_actions)
        
        mf_critic_loss = F.mse_loss(q_target, q_pred)
        self.optimizer_c.zero_grad()
        mf_critic_loss.backward()
        self.optimizer_c.step()
        self.soft_update_of_target_network(self.critic_local, self.critic_target, self.tau)
        
        return mf_critic_loss, mf_actor_loss

    def Auto_Transform(self,states,states_actions,next_states,rewards):
        q_pred = self.critic_local(states_actions)  # not sure here use local or target
        actions_pred = self.actor_local(states)
        q_target = rewards + self.gamma*self.critic_local(torch.cat((next_states, actions_pred), 1))
        diff = torch.abs(q_pred-q_target).view([-1])
        accuracy = diff / rewards
        accuracy_num = torch.sum(torch.lt(accuracy, self.c1))
        if accuracy_num > int(self.c2 * len(diff)):
            self.K += 1
            print(K)
            # here if I let K = K + 1, then those previously sampled tk_transition will be
            # automatically moved to Df, right?
