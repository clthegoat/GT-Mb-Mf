import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from joblib import Parallel, delayed
import random
import ilqr
import copy

from models.model_based import trans_model, reward_model, critic_model, actor_model

class OU_Noise(object):
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, seed=1, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.array(
            [np.random.normal() for _ in range(len(self.state))])
        self.state += dx
        return self.state


class OU_Noise_Exploration(object):
    """Ornstein-Uhlenbeck noise process exploration strategy"""
    def __init__(self, action_dim):
        self.noise = OU_Noise(action_dim)

    def perturb_action_for_exploration_purposes(self, action):
        """Perturbs the action of the agent to encourage exploration"""
        action += self.noise.sample()
        return action

    def reset(self):
        """Resets the noise process"""
        self.noise.reset()

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

    def sample(self, batch_size):
        return np.random.choice(self.memory[0:self.count], batch_size)


class MPC_agent():
    def __init__(self, conf):
        self.device = 'cpu'
        self.conf = conf
        self.dim_state = self.conf.data.state.dim
        self.dim_action = self.conf.data.action.dim
        self.dim_state_action = self.dim_state + self.dim_action
        self.trans_model = trans_model(self.dim_state, self.dim_action)
        #this is actually cost model
        self.reward_model = reward_model(self.dim_state, self.dim_action)
        self.critic_model = critic_model(self.dim_state, self.dim_action)
        self.target_critic = critic_model(self.dim_state, self.dim_action)
        #TODO: add policy network
        self.actor_model = actor_model(self.dim_state, self.dim_action)
        self.target_actor = actor_model(self.dim_state, self.dim_action)

        self.memory = Memory(self.conf.data.mem_capacity)
        self.optimizer_t = optim.Adam(self.trans_model.parameters(), lr=self.conf.train.mb_t_lr)
        self.optimizer_r = optim.Adam(self.reward_model.parameters(), lr=self.conf.train.mb_r_lr)
        self.optimizer_c = optim.Adam(self.critic_model.parameters(), lr=self.conf.train.mf_c_lr, 
                                    weight_decay = 0.01)
        self.optimizer_a = optim.Adam(self.actor_model.parameters(), lr=self.conf.train.mf_a_lr)
        self.trail_len = self.conf.train.trail_len
        self.mb_batch_size = self.conf.data.mb_mem_batchsize
        self.batch_size = self.conf.data.mem_batchsize

        #ilqr initial trajectory
        self.T = self.conf.planning.horizon
        self.gamma = self.conf.train.gamma

        #saved good trajectories for init
        self.saved_X = []
        self.saved_U = []

        #temporary solved planning trajectories
        self.traj_X = np.zeros((self.T + 1, self.dim_state, 1))
        self.traj_U = np.zeros((self.T, self.dim_action, 1))
        #init good trajectories
        for i in range(self.trail_len + 1):
            self.saved_X.append(self.traj_X)

        for i in range(self.trail_len):
            self.saved_U.append(self.traj_U)

        self.up_X = np.asarray([[1.], [1.], [8.]])
        self.low_X = -self.up_X
        self.up_U = 1.
        self.low_U = -self.up_U

        self.ilqr_lr = self.conf.planning.ilqr_learning_rate
        self.ilqr_iter_num = self.conf.planning.ilqr_iteration_num
        #ilqr controller
        self.K = np.zeros((self.T, self.dim_action, self.dim_state))
        self.k = np.zeros((self.T, self.dim_action, 1))

        self.num_random = self.conf.train.num_random
        self.fixed_num_per_reduction = self.conf.MBMF.fixed_num_per_reduction
        self.training_step = 0
        self.shooting_num = self.conf.planning.shooting_num
        self.tau = self.conf.MVE.target_model_update_rate

        self.exploration_strategy = OU_Noise_Exploration(self.dim_action)

        if self.conf.MBMF.reduction_type == "direct_fixed":
            self.backward = 0
        else:
            self.backward = 1

    def cost_model(self, state_action):

        return -self.reward_model(state_action)


    def value_model(self, state):
        '''
        given state, return the value function according to the current critic and actor
        input: state (torch)
        '''
        
        target_action = self.target_actor(state)
        value = -self.target_critic(torch.cat(
            (state, target_action), -1))[0]

        return value


    def select_action(self, state, exploration):
        '''
        use ilqr controller to select the action
        solve ilqr: update X,U in ctrl
        take first action for output
        num_step: which step the agent is on, used to select the initial traj
        state: current state
        

        exploration: whether add noise
        '''
        #get num of planning steps
        #print(result_action.shape)
        action = self.actor_model(state).cpu().data.numpy()

        if exploration:
            action = self.exploration_strategy.perturb_action_for_exploration_purposes(
                action)
            action = np.clip(action, self.low_U, self.up_U)

        return action

   

    def MB_target_compute(self, state, state_action, mode):
        '''
        given a state (torch), state_action, compute the action target and q target at one
        mode 0: random shooting
        mode 1: ilqr
        '''

        #policy init

        num_plan_step = self.T

        if mode == 0:
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

            critic_target = 0


        else:
       
            #state,action sequence
            X_seq = torch.zeros(num_plan_step + 1, 1, self.dim_state)
            U_seq = torch.zeros(num_plan_step, 1, self.dim_action)
            X_seq[0, :, :] = state.unsqueeze(0)
            
            #compute init traj
            for i in range(num_plan_step - 1):
                U_seq[i, :, :] = self.actor_model(X_seq[i, :, :])
                X_seq[i + 1, :, :] = self.trans_model(
                    torch.cat((X_seq[i, :, :], U_seq[i, :, :]), 1))


            X_seq = X_seq.reshape(
                (num_plan_step + 1, self.dim_state, 1)).cpu().detach().numpy()
            U_seq = U_seq.reshape(
                (num_plan_step, self.dim_action, 1)).cpu().detach().numpy()


            self.up_X = np.asarray([[1.], [1.], [8.]])
            self.low_X = -self.up_X
            self.up_U = 1.
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
            #print(critic_target)

        return target_action, critic_target

    def store_transition(self, transition):
        self.memory.update(transition)

    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data +
                                    (1.0 - tau) * target_param.data)


    def update(self, mode):
        '''
        mode 0: pre train (only trans and reward model)
        mode 1: normal train
        '''
        self.training_step += 1

        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'

        self.trans_model.to(device)
        self.reward_model.to(device)

        # data preparation
        if mode:
            transitions = self.memory.sample(self.mb_batch_size)
        else:
            transitions = self.memory.sample(self.batch_size)

        s = torch.from_numpy(
            np.vstack((t.s for t in transitions
                       if t is not None))).float().to(self.device)
        s_a = torch.tensor([t.s_a for t in transitions],
                           dtype=torch.float).view(-1, self.dim_state_action)
        s_ = torch.tensor([t.s_ for t in transitions],
                          dtype=torch.float).view(-1, self.dim_state)
        #actually train cost
        r = torch.tensor([t.r for t in transitions], dtype=torch.float).view(
            -1, 1)

        done = torch.tensor([t.done for t in transitions], dtype=torch.int).view(
            -1, 1)

        s_a, s_, r, done = s_a.to(device), s_.to(device), r.to(device), done.to(device)

        #get value target
        
        if mode:
            a_q_target_list = Parallel(n_jobs=4, prefer="threads")(
                delayed(self.MB_target_compute)(s_a[i][:self.dim_state], s_a[i], 0)
                for i in range(self.mb_batch_size))
            # a_q_target_list = [self.MB_target_compute(s[i], s_a[i],0) ##########adjust mode
            #     for i in range(self.batch_size)]
            a_target_list = [
                a_q_target_list[i][0] for i in range(self.mb_batch_size)
            ]
            q_target_list = [
                a_q_target_list[i][1] for i in range(self.mb_batch_size)
            ]
            
        

        # update transition model
        pred_state = self.trans_model(s_a)
        trans_loss = F.mse_loss(pred_state, s_)
        # print("transition loss: {}".format(trans_loss.item()))

        self.optimizer_t.zero_grad()
        trans_loss.backward()
        self.optimizer_t.step()

        # update reward model
        pred_reward = self.reward_model(s_a)
        reward_loss = F.mse_loss(pred_reward, r)
        # print("reward loss: {}".format(reward_loss.item()))
        self.optimizer_r.zero_grad()
        reward_loss.backward()
        self.optimizer_r.step()

       

        # update value model
        q_pred = self.critic_model(s_a).to(self.device)
        #q_target = torch.from_numpy(np.asarray(q_target_list)).float().to(
        #    self.device)
        s_a_target = torch.cat([s_,self.target_actor(s_)],1)

        with torch.no_grad():
            q_target = r + torch.mul(self.gamma*self.target_critic(s_a_target), 1-done)
        critic_loss = F.mse_loss(q_target, q_pred)
        self.optimizer_c.zero_grad()
        critic_loss.backward()
        self.optimizer_c.step()
        # print("critic loss: {}".format(critic_loss.item()))

        if mode:
            for g in self.optimizer_a.param_groups:
                g['lr'] = 1e-4
            # update action model
            a_pred = self.actor_model(s).to(self.device)
            with torch.no_grad():
                a_target = torch.from_numpy(np.asarray(a_target_list)).float().to(
                    self.device)
            actor_loss = F.mse_loss(a_target.view(-1,1), a_pred.view(-1,1))
            self.optimizer_a.zero_grad()
            actor_loss.backward()
            self.optimizer_a.step()
            # print("actor loss: {}".format(actor_loss.item()))

        else:
            for g in self.optimizer_a.param_groups:
                g['lr'] = 1e-4
            actions_pred = self.actor_model(s)
            actor_loss = -self.critic_model(
                torch.cat((s, actions_pred), 1)).to(
                    self.device).mean()
            self.optimizer_a.zero_grad()
            actor_loss.backward()
            self.optimizer_a.step()
            # print("actor loss: {}".format(actor_loss.item()))


        # update value target
        # TODO: change it to EMA?
        self.soft_update_of_target_network(self.critic_model,
                                        self.target_critic, self.tau)
        self.soft_update_of_target_network(self.actor_model,
                                        self.target_actor, self.tau)

        
        return trans_loss.item(), reward_loss.item(), actor_loss.item(), critic_loss.item()

    
