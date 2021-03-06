from numpy.core.defchararray import index
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from models.model_based import trans_model, reward_model, value_model, actor_model, critic_model, actor_time_model, critic_time_model
from MPC_agent import MPC_agent
from MVE_agent import MVE_agent
from collections import namedtuple

import ilqr
import joblib
from joblib import Parallel, delayed
import time


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
        '''
        sample a batch of data for model-based training
        '''
        memory = self.memory.tolist()
        mb_transition = namedtuple('mb_transition',
                                   ['s', 'a', 's_a', 's_', 'r', 't', 'done'])
        i = 0
        MB_memory = np.empty(self.capacity, dtype=object)
        for tran in memory:
            if (tran is not None) and (tran.t < (trail_len - K)):
                MB_memory[i] = mb_transition(tran.s, tran.a, tran.s_a, tran.s_,
                                             tran.r, tran.t, tran.done)
                i = i + 1
        new_MB_memory = MB_memory[np.s_[:i:1]]
        if i == 0:
            return []
        return np.random.choice(new_MB_memory[0:self.count], batch_size)

    def MF_sample(self, batch_size, trail_len, K):
        '''
        sample a batch of data for model-free training
        '''
        memory = self.memory.tolist()
        mf_transition = namedtuple('mf_transition',
                                   ['s', 'a', 's_a', 's_', 'r', 't', 'done'])
        i = 0
        MF_memory = np.empty(self.capacity, dtype=object)
        for tran in memory:
            if (tran is not None) and (tran.t >= (trail_len - K)):
                MF_memory[i] = mf_transition(tran.s, tran.a, tran.s_a, tran.s_,
                                             tran.r, tran.t, tran.done)
                i = i + 1
        new_MF_memory = MF_memory[np.s_[:i:1]]

        return np.random.choice(new_MF_memory[0:self.count], batch_size)

    def judge_sample(self, trail_len, K):
        memory = self.memory.tolist()
        mb_transition = namedtuple('mb_transition',
                                   ['s', 'a', 's_a', 's_', 'r', 't', 'done'])
        i = 0
        MB_memory = np.empty(int(self.capacity/2), dtype=object)
        for tran in memory:
            if (tran is not None) and (tran.t == (trail_len - K)):
                MB_memory[i] = mb_transition(tran.s, tran.a, tran.s_a, tran.s_,
                                             tran.r, tran.t, tran.done)
                i = i + 1
        new_MB_memory = MB_memory[np.s_[:i:1]]
        return new_MB_memory


class GTMBMF_agent(MVE_agent):
    def __init__(self, conf):
        super().__init__(conf)

        self.reduction_method = self.conf.GTMBMF.reduction_type
        self.memory = Memory(self.conf.data.mem_capacity)
        self.training_episode = -1
        self.T = self.conf.planning.horizon

        # here we temporarily fix K
        self.K = self.conf.train.K
        self.c1 = self.conf.train.c1
        self.c2 = self.conf.train.c2
        self.mb_batchsize = self.conf.data.mb_mem_batchsize

        #model based configs
        self.shooting_num = self.conf.planning.shooting_num
        self.ilqr_lr = self.conf.planning.ilqr_learning_rate
        self.ilqr_iter_num = self.conf.planning.ilqr_iteration_num
        self.plannning_method = self.conf.planning.method
        self.planning_mode = -1
        if self.plannning_method=="ilqr": #only shooting and ilqr
            self.planning_mode = 1
        if self.plannning_method=="shooting": #only shooting and ilqr
            self.planning_mode = 0


        ####first tried cpu
        self.device = 'cpu'
        self.time_dependent = False
        if self.time_dependent == True:
            self.critic_local = critic_time_model(
                self.dim_state, self.dim_action).to(self.device)
            self.critic_target = critic_time_model(
                self.dim_state, self.dim_action).to(self.device)
            self.copy_model_over(self.critic_local, self.critic_target)
            self.actor_local = actor_time_model(
                self.dim_state, self.dim_action).to(self.device)
            self.actor_target = actor_time_model(
                self.dim_state, self.dim_action).to(self.device)
            self.copy_model_over(self.actor_local, self.actor_target)
        self.optimizer_mb_c = optim.Adam(self.critic_local.parameters(),
                                         lr=self.conf.train.mb_c_lr)
        self.optimizer_mb_a = optim.Adam(self.actor_local.parameters(),
                                         lr=self.conf.train.mb_a_lr)

        #reduction
        self.fixed_num_per_reduction = self.conf.GTMBMF.fixed_num_per_reduction
        if self.conf.GTMBMF.reduction_type == "direct_fixed":
            self.backward = 0
        else:
            self.backward = 1

        self.tmp_training_episode = 0

    '''
    No value model here, so build it based on critic model and target actor
    '''
    def value_model(self, state):
        '''
        given state, return the value function according to the current critic and actor
        input: state (torch)
        '''
        target_action = self.actor_target(state)
        value = -self.critic_target(torch.cat(
            (state, target_action), -1))[0]
        return value

    def cost_model(self, state_action):

        return -self.reward_model(state_action)

    def gtmbmf_select_action(self, num_step, state, exploration, relative_step):
        '''
        according to yunkao, there are two choices for selecting action
        1) select actions w.r.t step
        2) always use MF to select action
        :return:
        input: state (numpy?)
        '''

        # first choice
        if relative_step == 0:
            print(num_step)
            if num_step < self.trail_len - self.K:
                #model based
                #random shooting
                num_plan_step = min(
                    [self.T, self.trail_len - self.K - num_step])
                X_0 = state.reshape((-1, 1))
                min_c = 1000000
                for i in range(self.shooting_num):
                    U = np.random.uniform(self.low_U, self.up_U,
                                          (num_plan_step, self.dim_action, 1))
                    X, c = ilqr.forward_sim(X_0, U, self.trans_model,
                                            self.cost_model, self.value_model)
                    if c < min_c:
                        min_c = c
                        min_U = U
                        min_X = X
                # X_seq = torch.zeros(num_plan_step+1,1,self.dim_state)
                # U_seq = torch.zeros(num_plan_step,1,self.dim_action)
                # X_seq[0,:,:] = state.unsqueeze(0)
                # for i in range(num_plan_step-1):
                #     # U_seq[i,:,:] = self.actor_local(X_seq[i,:,:])
                #     U_seq[i,:,:] = self.actor_target(X_seq[i,:,:])
                #     X_seq[i+1,:,:] = self.trans_model(torch.cat((X_seq[i,:,:], U_seq[i,:,:]), 1))
                # X_seq = X_seq.reshape((num_plan_step+1,self.dim_state,1)).cpu().detach().numpy()
                # U_seq = U_seq.reshape((num_plan_step,self.dim_action,1)).cpu().detach().numpy()

                #do ilqr based on best one
                ilqr_ctrl = ilqr.ilqr_controller(
                    min_X, min_U, self.up_X, self.low_X, self.up_U, self.low_U,
                    num_plan_step, self.trans_model, self.cost_model,
                    self.value_model, self.ilqr_lr, self.ilqr_iter_num, 0, 0)

                _, _, traj_X, traj_U, _, _ = ilqr_ctrl.solve_ilqr()

                return traj_U[0, :, :].reshape((-1, ))

            else:
                #model free
                state = state.unsqueeze(0).to(self.device)
                return self.actor_local(state).cpu().data.numpy()
        else:
            # second choice
            state = state.unsqueeze(0).to(self.device)
            if self.time_dependent == True:
                action = self.actor_local(torch.cat((state, num_step),
                                                    1)).cpu().data.numpy()
            else:
                action = self.actor_local(state).cpu().data.numpy()
            if exploration:
                action = self.exploration_strategy.perturb_action_for_exploration_purposes(
                    action)
                #action = action + np.random.normal(0, 0.2, self.dim_action)
                action = np.clip(action, -1., 1.)
            return action

    def store_transition(self, transition):
        return self.memory.update(transition)

    def sample_transitions(self, sampleType):
        if sampleType == "all":
            transitions = self.memory.all_sample(batch_size=self.batch_size)
        elif sampleType == "MB":
            transitions = self.memory.MB_sample(batch_size=self.mb_batchsize,
                                                trail_len=self.trail_len,
                                                K=self.K)

        elif sampleType == "MF":
            transitions = self.memory.MF_sample(batch_size=self.batch_size,
                                                trail_len=self.trail_len,
                                                K=self.K)
        elif sampleType == "judge":
            transitions = self.memory.judge_sample(trail_len=self.trail_len,
                                                   K=self.K)
        else:
            raise ValueError
        #if MB is empty
        #if it is empty
        if len(transitions) == 0:
            return None, None, None, None, None, None

        s = torch.from_numpy(
            np.vstack((t.s for t in transitions
                       if t is not None))).float().to(self.device)
        a = torch.from_numpy(
            np.vstack((t.a.reshape((1,-1)) for t in transitions
                       if t is not None))).float().to(self.device)
        s_a = torch.from_numpy(
            np.vstack((t.s_a.reshape((1,-1)) for t in transitions
                       if t is not None))).float().to(self.device)
        s_ = torch.tensor([t.s_ for t in transitions], dtype=torch.float).view(
            -1, self.dim_state).to(self.device)
        r = torch.tensor([t.r for t in transitions],
                         dtype=torch.float).view(-1, 1).to(self.device)
        t = torch.tensor([t.t for t in transitions],
                         dtype=torch.float).view(-1, 1).to(self.device)

        dones = torch.tensor([t.done for t in transitions],
                         dtype=torch.int).view(-1, 1).to(self.device)

        return s, a, s_a, s_, r, t,  dones

    def update(self, mode):
        '''
        mode=0:
        only update reward, trans model
        mode=1:
        update all
        '''
        if self.reduction_method == "back_fixed":
            # print("GTMBMF auto reduction method: back fixed")
            self.backward = True
            self.critiria = False
        elif self.reduction_method == "direct_fixed":
            self.backward = False
            self.critiria = False
            # print("GTMBMF auto reduction method: direct fixed")
        elif self.reduction_method == "back_bellman":
            # print("GTMBMF auto reduction method: back bellman equation")
            self.backward = False
            self.critiria = True
        else:
            # print("Please indicate GTMBMF auto reduction method! Now default: back bellman equation")
            self.backward = False
            self.critiria = True
        
        self.training_step += 1
        """ update transition and reward model, data sampled from all memory"""
        _, _, all_s_a, all_s_, all_r, _, _ = self.sample_transitions("all")

        # update transition model
        trans_loss = self.trans_learn(all_s_a, all_s_)

        # update reward model
        reward_loss = self.reward_learn(all_s_a, all_r)

        mb_critic_loss, mb_actor_loss, mf_actor_loss, mf_critic_loss = 0.0, 0.0, 0.0, 0.0
        if self.critiria == True:
            if (mode==1 and self.T>0):
                """ update critic and actor model, data sampled from MB memory"""
                mb_s, _, mb_s_a, mb_s_, mb_r, mb_t, mb_d = self.sample_transitions("MB")
                if not (mb_s == None):
                    mb_critic_loss, mb_actor_loss = self.MB_learn(
                        mb_s, mb_s_a, mb_s_, mb_r, mb_t, mb_d)
                """ automatic transformation"""
                tk_s, _, tk_s_a, tk_s_, tk_r, tk_t = self.sample_transitions(
                    "judge")
                if not tk_s == None:
                    self.Auto_Transform(tk_s, tk_s_a, tk_s_, tk_r, tk_t)
            else:
                """ update critic and actor model, data sampled from MF memory"""
                mf_s, _, mf_s_a, mf_s_, mf_r, mb_t, mf_d = self.sample_transitions("all")
                mf_critic_loss, mf_actor_loss = self.MF_learn(mf_s, mf_s_a, mf_s_,
                                                            mf_r, mb_t, mf_d)
            return trans_loss, reward_loss, mb_actor_loss, mb_critic_loss, mf_actor_loss, mf_critic_loss
        else:
            if (mode and not self.backward and self.T>0) or (mode and self.backward and self.K!=self.trail_len):
                """ update critic and actor model, data sampled from MB memory"""
                mb_s, _, mb_s_a, mb_s_, mb_r, mb_t, mb_d = self.sample_transitions("MB")
                if not (mb_s == None):
                    mb_critic_loss, mb_actor_loss = self.MB_learn(
                        mb_s, mb_s_a, mb_s_, mb_r, mb_t, mb_d)
                """ fixed transformation"""
                if ((self.training_episode - self.num_random) % self.fixed_num_per_reduction==0) and not self.tmp_training_episode == self.training_episode:
                    
                    if self.backward:
                        self.K += 1
                    else:
                        self.T = max(self.T-1,0)
                    self.tmp_training_episode = self.training_episode
            else:
                """ update critic and actor model, data sampled from MF memory"""
                mf_s, _, mf_s_a, mf_s_, mf_r, mb_t, mf_d = self.sample_transitions("all")
                mf_critic_loss, mf_actor_loss = self.MF_learn(mf_s, mf_s_a, mf_s_,
                                                            mf_r, mb_t, mf_d)
            return trans_loss, reward_loss, mb_actor_loss, mb_critic_loss, mf_actor_loss, mf_critic_loss

    def MB_target_compute(self, state, state_action, next_state, reward, time_step, done, mode):
        '''
        given a state (torch), state_action, compute the action target and q target at one
        mode 0: random shooting
        mode 1: ilqr
        '''

        #     time_step = time_steps[i]
        #     cur_state = states[i]
        #     j = 0
        #     while time_step < self.trail_len - self.K:  # here not sure whether < or <=
        #         action = torch.tensor(self.gtmbmf_select_action(time_step,\
        #                                                       cur_state,\
        #                                                       relative_step=j,\
        #                                                       mode=2,\
        #                                                       exploration=0), dtype=torch.float).to(self.device)
        #         state_action = torch.cat((cur_state, action))
        #         # record action (only record target action and predicted action on first step)
        #         if j==0:
        #             a_target[i] = action
        #             a_pred[i] = self.actor_local(cur_state)

        #         reward = self.reward_model(state_action)
        #         cur_state = self.trans_model(state_action)
        #         q_target[i:i+1] += reward * self.gamma**j
        #         time_step += 1
        #         j += 1

        #     action_H = torch.tensor(self.gtmbmf_select_action(time_step, \
        #                                                     cur_state, \
        #                                                     relative_step=j, \
        #                                                     mode=2, \
        #                                                     exploration=0), dtype=torch.float).to(self.device)
        #     state_action_H = torch.cat((cur_state, action_H))
        #     with torch.no_grad():
        #         q_target[i:i+1] += self.critic_target(state_action_H) * self.gamma ** j

        #random shooting
        if mode:
            time_step = np.int(time_step.cpu().detach().numpy())
            if self.backward:
                num_plan_step = min([self.T, self.trail_len - self.K - time_step])
            else:
                num_plan_step = self.T
            ## old version: use random shooting for initialization
            # X_0 = state.cpu().detach().numpy().reshape((-1,1))
            # min_c = 1000000
            # for i in range(self.shooting_num):
            #     #print(num_plan_step)
            #     U = np.random.uniform(self.low_U,self.up_U,(num_plan_step,self.dim_action,1))
            #     X, c = ilqr.forward_sim(X_0,U,self.trans_model,self.cost_model,self.value_model)
            #     if c<min_c:
            #         min_c = c
            #         min_U = U
            #         min_X = X
            # X_seq = min_X
            # U_seq = min_U
            ## new version: use learned policy for initialization
            X_seq = torch.zeros(num_plan_step + 1, 1, self.dim_state)
            U_seq = torch.zeros(num_plan_step, 1, self.dim_action)
            X_seq[0, :, :] = state.unsqueeze(0)
            if self.time_dependent == True:
                for i in range(num_plan_step - 1):
                    U_seq[i, :, :] = self.actor_local(
                        torch.cat((X_seq[i, :, :],
                                torch.Tensor(time_step + i).unsqueeze(0)), 1))
                    X_seq[i + 1, :, :] = self.trans_model(
                        torch.cat((X_seq[i, :, :], U_seq[i, :, :]), 1))
            else:
                for i in range(num_plan_step - 1):
                    U_seq[i, :, :] = self.actor_local(X_seq[i, :, :])
                    X_seq[i + 1, :, :] = self.trans_model(
                        torch.cat((X_seq[i, :, :], U_seq[i, :, :]), 1))
            X_seq = X_seq.reshape(
                (num_plan_step + 1, self.dim_state, 1)).cpu().detach().numpy()
            U_seq = U_seq.reshape(
                (num_plan_step, self.dim_action, 1)).cpu().detach().numpy()

            self.up_X = np.asarray([[1.], [1.], [8.]])
            self.low_X = -self.up_X
            self.up_U = 2.
            self.low_U = -self.up_U
            #do ilqr based on best one
            ilqr_ctrl = ilqr.ilqr_controller(X_seq, U_seq, self.up_X, self.low_X,
                                            self.up_U, self.low_U, num_plan_step,
                                            self.trans_model, self.cost_model,
                                            self.value_model, self.ilqr_lr,
                                            self.ilqr_iter_num, 0, 0)

            _, _, traj_X, traj_U, C, last_value = ilqr_ctrl.solve_ilqr()

            #compute target action
            target_action = traj_U[0, :, :]  #m*1

            #compute target value
            gamma_array = np.logspace(0,
                                    num_plan_step,
                                    num_plan_step + 1,
                                    base=self.gamma)
            R_value = np.concatenate([-C, -last_value.reshape((-1, ))])

            critic_target = np.dot(gamma_array, R_value)
        else:
            num_plan_step = self.T
            X_seq = torch.zeros(num_plan_step + 1, self.shooting_num, self.dim_state)
            batch_cost = torch.zeros(self.shooting_num, 1)
            U = np.random.uniform(-1., 1.,
                                    (num_plan_step, self.shooting_num, self.dim_action))
            U_seq = torch.from_numpy(U).float().to(self.device)
            X_seq[0, :, :] = state.unsqueeze(0)
            
            for i in range(num_plan_step - 1):
                X_seq[i + 1, :, :] = self.trans_model(
                    torch.cat((X_seq[i, :, :], U_seq[i, :, :]), 1))

                batch_cost += self.cost_model(
                    torch.cat((X_seq[i, :, :], U_seq[i, :, :]), 1)
                )

            batch_cost += self.value_model(X_seq[-1, :, :])


            X_seq = X_seq.reshape(
                (num_plan_step + 1, self.shooting_num, self.dim_state)).cpu().detach().numpy()
            U_seq = U_seq.reshape(
                (num_plan_step, self.shooting_num, self.dim_action)).cpu().detach().numpy()

            min_c, index = torch.min(batch_cost, 0)

            min_U = U_seq[:, index, :]

            target_action = min_U[0]

            action_pred = self.actor_target(next_state)
            critic_target = reward + torch.mul(self.gamma * self.critic_target(
                torch.cat((next_state, action_pred), -1)), 1-done).cpu().detach().numpy()

        return target_action, critic_target

    def MB_learn(self, states, states_actions, next_states, rewards, time_steps, dones):
        '''
        train actor&critic net through model-based method
        '''
        # q prediction and target
        if self.time_dependent == True:
            q_pred = self.critic_local(
                torch.cat((states_actions, time_steps), 1)).to(self.device)
        else:
            q_pred = self.critic_local(states_actions).to(self.device)
        q_target = torch.zeros(self.mb_batchsize).to(self.device)

        # policy prediction and target
        a_target = torch.zeros(self.mb_batchsize).to(self.device)
        a_pred = torch.zeros(self.mb_batchsize).to(self.device)

        #compute batch targets
        #print(joblib.cpu_count())
        #time_start = time.clock()
        with torch.no_grad():
            a_q_target_list = Parallel(n_jobs=4, prefer="threads")(
                delayed(self.MB_target_compute)(states[i], states_actions[i], 
                                                next_states[i],
                                                rewards[i], 
                                                time_steps[i],
                                                dones[i],
                                                self.planning_mode)
                for i in range(self.mb_batchsize))
            
            a_target_list = [
                a_q_target_list[i][0] for i in range(self.mb_batchsize)
            ]
            q_target_list = [
                a_q_target_list[i][1] for i in range(self.mb_batchsize)
            ]
            a_target = torch.from_numpy(np.asarray(a_target_list)).float().to(
                self.device)
            q_target = torch.from_numpy(np.asarray(q_target_list)).float().to(
                self.device)


        if self.time_dependent == True:
            a_pred = self.actor_local(torch.cat((states, time_steps),
                                                1)).to(self.device)
            q_pred = self.critic_local(
                torch.cat((states_actions, time_steps), 1)).to(self.device)
        else:
            a_pred = self.actor_local(states).to(self.device)
            q_pred = self.critic_local(states_actions).to(self.device)

        # update mb-critic model
        q_target, q_pred = q_target.view([-1, 1]), q_pred.view([-1, 1])
        mb_critic_loss = F.mse_loss(q_target, q_pred)
        self.optimizer_mb_c.zero_grad()
        mb_critic_loss.backward()
        self.optimizer_mb_c.step()
        self.soft_update_of_target_network(self.critic_local,
                                           self.critic_target, self.tau)

        # update mb-actor model
        a_target, a_pred = a_target.view([-1, 1]), a_pred.view([-1, 1])
        mb_actor_loss = F.mse_loss(a_pred, a_target)
        self.optimizer_mb_a.zero_grad()
        mb_actor_loss.backward()
        self.optimizer_mb_a.step()
        self.soft_update_of_target_network(self.actor_local, self.actor_target,
                                           self.tau)

        return mb_critic_loss, mb_actor_loss

    def MF_learn(self, states, states_actions, next_states, rewards,
                 time_steps, dones):
        '''
        train actor&critic net through model-free method
        '''
        if self.time_dependent == True:
            mf_actor_loss = self.actor_learn(states, time_steps)
            actions_pred = self.actor_target(
                torch.cat((next_states, time_steps + 1), 1))
            q_target = rewards + self.gamma * self.critic_target(
                torch.cat((next_states, actions_pred, time_steps + 1), 1))
            q_pred = self.critic_local(
                torch.cat((states_actions, time_steps), 1))
        else:
            # update mf-actor model
            mf_actor_loss = self.actor_learn(states)
            # update mf-critic model
            actions_pred = self.actor_target(next_states)
            with torch.no_grad():
                q_target = rewards + torch.mul(self.gamma * self.critic_target(
                    torch.cat((next_states, actions_pred), 1)),
                    1-dones)
            q_pred = self.critic_local(states_actions)

        mf_critic_loss = F.mse_loss(q_target, q_pred)
        self.optimizer_c.zero_grad()
        mf_critic_loss.backward()
        self.optimizer_c.step()
        self.soft_update_of_target_network(self.critic_local,
                                           self.critic_target, self.tau)

        return mf_critic_loss, mf_actor_loss

    def actor_learn(self, states, time_steps=None):
        """Runs a learning iteration for the actor"""
        if time_steps == None:
            actions_pred = self.actor_local(states).to(self.device)
            actor_loss = -self.critic_local(
                torch.cat((states, actions_pred), 1)).to(self.device).mean()
        else:
            actions_pred = self.actor_local(torch.cat((states, time_steps),
                                                      1)).to(self.device)
            actor_loss = -self.critic_local(
                torch.cat((states, actions_pred, time_steps), 1)).to(
                    self.device).mean()
        self.optimizer_a.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(),
                                 self.max_grad_norm)
        self.optimizer_a.step()
        self.soft_update_of_target_network(self.actor_local, self.actor_target,
                                           self.tau)

        return actor_loss.item()

    def Auto_Transform(self, states, states_actions, next_states, rewards,
                       time_steps):
        if self.time_dependent == True:
            q_pred = self.critic_local(
                torch.cat((states_actions, time_steps),
                          1))  
            actions_pred = self.actor_local(
                torch.cat((next_states, time_steps + 1), 1))
            q_target = rewards + self.gamma * self.critic_local(
                torch.cat((next_states, actions_pred, time_steps + 1), 1))
        else:
            q_pred = self.critic_local(
                states_actions)  
            actions_pred = self.actor_local(next_states)
            q_target = rewards + self.gamma * self.critic_local(
                torch.cat((next_states, actions_pred), 1))
        diff = torch.abs(q_pred - q_target)
        #use torch.div
        err = torch.div(diff, torch.abs(q_target))
        err_num = torch.sum(torch.lt(err, self.c1))
        print(float(err_num) / diff.size(0))
        if err_num > int(self.c2 * diff.size(0)):
            self.K += 1
            # here if I let K = K + 1, then those previously sampled tk_transition will be
            # automatically moved to Df, right?
