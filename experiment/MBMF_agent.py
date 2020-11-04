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

    def MB_sample(self, batch_size, trail_len, K):
        memory = self.memory.tolist()
        mb_transition = namedtuple('mb_transition', ['s', 'a', 's_a', 's_', 'r', 't', 'done'])
        i = 0
        MB_memory = np.empty(1000, dtype=object)
        for tran in memory:
            if (tran is not None) and (tran.t < (trail_len - K)):
                MB_memory[i] = mb_transition(tran.s, tran.a, tran.s_a, tran.s_, tran.r, tran.t, tran.done)
                i = i + 1
        new_MB_memory = MB_memory[np.s_[:i:1]]
        return np.random.choice(new_MB_memory[0:self.count], batch_size)

    def MF_sample(self, batch_size, trail_len, K):
        memory = self.memory.tolist()
        mb_transition = namedtuple('mb_transition', ['s', 'a', 's_a', 's_', 'r', 't', 'done'])
        i = 0
        MB_memory = np.empty(1000, dtype=object)
        for tran in memory:
            if (tran is not None) and (tran.t >= (trail_len - K)):
                MB_memory[i] = mb_transition(tran.s, tran.a, tran.s_a, tran.s_, tran.r, tran.t, tran.done)
                i = i + 1
        new_MB_memory = MB_memory[np.s_[:i:1]]
        return np.random.choice(new_MB_memory[0:self.count], batch_size)

    def judge_sample(self, trail_len, K):
        # memory = self.memory.tolist()
        # trans_memory = [tran for tran in memory if tran.t == trail_len - K]
        # trans_memory = np.array(trans_memory)
        # return trans_memory
        memory = self.memory.tolist()
        mb_transition = namedtuple('mb_transition', ['s', 'a', 's_a', 's_', 'r', 't', 'done'])
        i = 0
        MB_memory = np.empty(1000, dtype=object)
        for tran in memory:
            if (tran is not None) and (tran.t == (trail_len - K)):
                MB_memory[i] = mb_transition(tran.s, tran.a, tran.s_a, tran.s_, tran.r, tran.t, tran.done)
                i = i + 1
        new_MB_memory = MB_memory[np.s_[:i:1]]
        return new_MB_memory
        # return np.random.choice(new_MB_memory[0:self.count], batch_size)


class MBMF_agent(MVE_agent):

    def __init__(self, conf):
        super().__init__(conf)

        self.memory = Memory(self.conf.data.mem_capacity)

        # agent
        self.mb_agent = MPC_agent(self.conf)
        self.mf_agent = MVE_agent(self.conf)

        # here we temporarily fix K
        self.K = self.conf.train.K
        self.c1 = self.conf.train.c1
        self.c2 = self.conf.train.c2
        self.mb_batchsize = self.conf.data.mb_mem_batchsize
        # self.trail_len = self.conf.trail_len # steps in each trail
        # self.batch_size = self.conf.data.mem_batchsize

    def mbmf_select_action(self, num_step, state, mode, exploration, relative_step):
        '''
        according to yunkao, there are two choices for selecting action
        1) select actions w.r.t step
        2) always use MF to select action
        :return:
        '''

        # first choice
        if relative_step == 0:
            if num_step <= self.trail_len - self.K:
                return self.mb_agent.select_action(num_step, state, mode, exploration)
            else:
                return self.mf_agent.select_action(state, exploration)
        else:
        # second choice
            return self.mf_agent.select_action(state, exploration)

    def store_transition(self, transition):
        return self.memory.update(transition)

    def sample_transitions(self, sampleType):
        if sampleType == "all":
            transitions = self.memory.all_sample(batch_size=self.batch_size)
        elif sampleType == "MB":
            transitions = self.memory.MB_sample(batch_size=self.mb_batchsize, trail_len=self.trail_len, K=self.K)
        elif sampleType == "MF":
            transitions = self.memory.MF_sample(batch_size=self.batch_size, trail_len=self.trail_len, K=self.K)
        elif sampleType == "judge":
            transitions = self.memory.judge_sample(trail_len=self.trail_len, K=self.K)
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
        mb_s, mb_s_a, mb_t = mb_s.to(self.device), mb_s_a.to(self.device), mb_t.to(self.device)
        mb_critic_loss, mb_actor_loss = self.MB_learn(mb_s, mb_s_a, mb_t)
        print("MB actor loss: {}".format(mb_actor_loss))
        print("MB critic loss: {}".format(mb_critic_loss))

        """ update critic and actor model, data sampled from MF memory"""
        mf_s, _, mf_s_a, mf_s_, mf_r, _ = self.sample_transitions("MF")
        mf_s, mf_s_a, mf_s_, mf_r = mf_s.to(self.device), mf_s_a.to(self.device), mf_s_.to(self.device), mf_r.to(self.device)
        mf_critic_loss, mf_actor_loss = self.MF_learn(mf_s, mf_s_a, mf_s_, mf_r)
        print("MF actor loss: {}".format(mf_actor_loss))
        print("MF critic loss: {}".format(mf_critic_loss))

        """ automatic transformation"""
        tk_s, _, tk_s_a, tk_s_, tk_r, _ = self.sample_transitions("judge")
        tk_s, tk_s_a, tk_s_, tk_r = tk_s.to(self.device), tk_s_a.to(self.device), tk_s_.to(self.device), tk_r.to(self.device)
        self.Auto_Transform(tk_s, tk_s_a, tk_s_, tk_r)
        
    
    def MB_learn(self,states,states_actions,time_steps):
        # q prediction and target
        q_pred = self.critic_local(states_actions)
        q_target = torch.zeros(self.mb_batchsize).to(self.device)

        # policy prediction and target
        a_target = torch.zeros(self.mb_batchsize)
        a_pred = torch.zeros(self.mb_batchsize)

        for i in range(self.mb_batchsize):
            time_step = time_steps[i]
            cur_state = states[i]
            j = 0
            while time_step < self.trail_len - self.K:  # here not sure whether < or <=
                action = torch.tensor(self.mbmf_select_action(time_step,\
                                                              cur_state,\
                                                              relative_step=j,\
                                                              mode=2,\
                                                              exploration=0), dtype=torch.float).to(self.device)
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

            action_H = torch.tensor(self.mbmf_select_action(time_step, \
                                                            cur_state, \
                                                            relative_step=j, \
                                                            mode=2, \
                                                            exploration=0), dtype=torch.float).to(self.device)
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
            # here if I let K = K + 1, then those previously sampled tk_transition will be
            # automatically moved to Df, right?
