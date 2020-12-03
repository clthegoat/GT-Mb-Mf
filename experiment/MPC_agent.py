import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import ilqr

from models.model_based import trans_model, reward_model, value_model


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
        self.conf = conf
        self.dim_state = self.conf.data.state.dim
        self.dim_action = self.conf.data.action.dim
        self.dim_state_action = self.dim_state + self.dim_action
        self.trans_model = trans_model(self.dim_state, self.dim_action)
        #this is actually cost model
        self.reward_model = reward_model(self.dim_state, self.dim_action)
        self.value_model = value_model(self.dim_state)
        self.target_value = value_model(self.dim_state)

        self.memory = Memory(self.conf.data.mem_capacity)
        self.optimizer_t = optim.Adam(self.trans_model.parameters(), lr=1e-3)
        self.optimizer_r = optim.Adam(self.reward_model.parameters(), lr=1e-3)
        self.optimizer_v = optim.Adam(self.value_model.parameters(), lr=1e-3)
        self.trail_len = self.conf.train.trail_len
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
        self.up_U = 2.
        self.low_U = -self.up_U

        self.ilqr_lr = self.conf.planning.ilqr_learning_rate
        self.ilqr_iter_num = self.conf.planning.ilqr_iteration_num
        #ilqr controller
        self.K = np.zeros((self.T, self.dim_action, self.dim_state))
        self.k = np.zeros((self.T, self.dim_action, 1))

        self.num_random = self.conf.train.num_random
        self.action_noise = self.conf.train.action_noise

        self.shooting_num = self.conf.planning.shooting_num

        self.training_step = 0
        self.target_update = self.conf.train.target_update_num

    def select_action(self, num_step, state, mode, exploration):
        '''
        use ilqr controller to select the action
        solve ilqr: update X,U in ctrl
        take first action for output
        num_step: which step the agent is on, used to select the initial traj
        state: current state
        mode: action choosing mode:
            0: fixed initial state, ilqr + mpc
            1: pure ilqr
            2: random shooting + ilqr + mpc
            3: ilqr + mpc, initialize with traj of last step

        exploration: whether add noise
        '''
        #get num of planning steps
        num_plan_step = min([self.T, self.trail_len - num_step])

        #only receding horizon control, fixed initial
        if mode == 0:

            #initialize with previous saved trajecotries
            #therefore also linearize around previous saved real interaction trajecories(not planned trajectories)
            #may cause problem when model is not accurate?
            self.traj_U = self.saved_U[num_step][0:num_plan_step]
            X_0 = state.detach().numpy().reshape((-1, 1))
            self.traj_X, _ = ilqr.forward_sim(X_0, self.traj_U,
                                              self.trans_model,
                                              self.reward_model,
                                              self.value_model)

            ilqr_ctrl = ilqr.ilqr_controller(
                self.traj_X, self.traj_U, self.up_X, self.low_X, self.up_U,
                self.low_U, num_plan_step, self.trans_model, self.reward_model,
                self.value_model, self.ilqr_lr, self.ilqr_iter_num, 0, 0)

            self.K, self.k, traj_X, traj_U, _, _ = ilqr_ctrl.solve_ilqr()
            self.traj_X = traj_X
            self.traj_U = traj_U

            #save interaction data for initialization of later episodes
            self.saved_X[num_step] = self.traj_X
            self.saved_U[num_step] = self.traj_U

            result_action = self.traj_U[0, :, :].reshape((-1, ))

        #pure ilqr, only compute at the first step
        #when using mode 1, num_plan_step shoud equals to trail length
        elif mode == 1:
            if num_step == 0:
                #initialize with previous saved trajecotries
                #therefore also linearize around previous saved real interaction trajecories(not planned trajectories)
                #may cause problem when model is not accurate?
                self.traj_U = self.saved_U[num_step][0:num_plan_step]
                X_0 = state.detach().numpy().reshape((-1, 1))
                self.traj_X, _ = ilqr.forward_sim(X_0, self.traj_U,
                                                  self.trans_model,
                                                  self.reward_model,
                                                  self.value_model)

                ilqr_ctrl = ilqr.ilqr_controller(
                    self.traj_X, self.traj_U, self.up_X, self.low_X, self.up_U,
                    self.low_U, num_plan_step, self.trans_model,
                    self.reward_model, self.value_model, self.ilqr_lr,
                    self.ilqr_iter_num, 0, 0)

                self.K, self.k, traj_X, traj_U, _, _ = ilqr_ctrl.solve_ilqr()

                self.traj_X = traj_X
                self.traj_U = traj_U

                #save interaction data for initialization of later episodes
                self.saved_X[num_step] = self.traj_X
                self.saved_U[num_step] = self.traj_U

                result_action = self.traj_U[0, :, :].reshape((-1, ))

            else:
                dx = state.reshape((-1, 1)) - self.traj_X[num_step, :, :]
                result_action = self.ilqr_lr * (np.dot(
                    self.K[num_step, :, :], dx) + self.k[num_step, :, :]) + (
                        1 - self.ilqr_lr) * self.traj_U[num_step, :, :]
                result_action = result_action.reshape((-1, ))

        elif mode == 2:
            #random shooting:

            X_0 = state.cpu().detach().numpy().reshape((-1, 1))
            min_c = 1000000
            for i in range(self.shooting_num):
                U = np.random.uniform(self.low_U, self.up_U,
                                      (num_plan_step, self.dim_action, 1))
                X, c = ilqr.forward_sim(X_0, U, self.trans_model,
                                        self.reward_model, self.value_model)
                if c < min_c:
                    min_c = c
                    min_U = U
                    min_X = X

            #do ilqr based on best one

            ilqr_ctrl = ilqr.ilqr_controller(min_X, min_U, self.up_X,
                                             self.low_X, self.up_U, self.low_U,
                                             num_plan_step, self.trans_model,
                                             self.reward_model,
                                             self.value_model, self.ilqr_lr,
                                             self.ilqr_iter_num, 0, 0)

            self.K, self.k, traj_X, traj_U, _, _ = ilqr_ctrl.solve_ilqr()

            self.traj_X = traj_X
            self.traj_U = traj_U
            # self.traj_U = min_U
            # self.traj_X = min_X

            result_action = self.traj_U[0, :, :].reshape((-1, ))

        elif mode == 3:
            #TODO: add warm up based on last time step
            #random shooting:
            if num_step == 0:
                X_0 = state.detach().numpy().reshape((-1, 1))
                min_c = 1000000
                for i in range(self.shooting_num):
                    U = np.random.uniform(self.low_U, self.up_U,
                                          (num_plan_step, self.dim_action, 1))
                    X, c = ilqr.forward_sim(X_0, U, self.trans_model,
                                            self.reward_model,
                                            self.value_model)
                    if c < min_c:
                        min_c = c
                        min_U = U
                        min_X = X

            else:
                #extract the last actions from previous optimization for initialization
                min_U = self.traj_U[1:min(num_plan_step + 1, self.T), :, :]
                X_0 = state.detach().numpy().reshape((-1, 1))
                min_X, c = ilqr.forward_sim(X_0, min_U, self.trans_model,
                                            self.reward_model,
                                            self.value_model)
                min_v = 100000
                if num_plan_step == self.T:
                    #append a new action at the end with random shooting
                    for i in range(self.shooting_num):
                        u = np.random.uniform(self.low_U, self.up_U,
                                              (1, self.dim_action, 1))
                        xu = torch.from_numpy(
                            np.concatenate([min_X[-1, :, :],
                                            u[0, :, :]])[:, 0]).float()
                        new_x = self.trans_model(xu)
                        v = self.reward_model(
                            xu) + self.gamma * self.value_model(new_x)
                        if v < min_v:
                            min_v = v
                            min_u = u
                            min_x = new_x.detach().numpy().reshape(
                                (1, self.dim_state, 1))

                    min_U = np.concatenate([min_U, min_u])
                    min_X = np.concatenate([min_X, min_x])

            #do ilqr based on best one

            ilqr_ctrl = ilqr.ilqr_controller(min_X, min_U, self.up_X,
                                             self.low_X, self.up_U, self.low_U,
                                             num_plan_step, self.trans_model,
                                             self.reward_model,
                                             self.value_model, self.ilqr_lr,
                                             self.ilqr_iter_num, 0, 0)

            self.K, self.k, traj_X, traj_U, _, _ = ilqr_ctrl.solve_ilqr()

            self.traj_X = traj_X
            self.traj_U = traj_U
            # self.traj_U = min_U
            # self.traj_X = min_X

            result_action = self.traj_U[0, :, :].reshape((-1, ))

            #print(result_action.shape)
        return result_action + exploration * np.random.normal(
            0.0, self.action_noise, (self.dim_action))

    def store_transition(self, transition):
        self.memory.update(transition)

    def update(self):
        self.training_step += 1

        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'

        self.trans_model.to(device)
        self.reward_model.to(device)

        # data preparation
        transitions = self.memory.sample(self.conf.data.mem_batchsize)
        s_a = torch.tensor([t.s_a for t in transitions],
                           dtype=torch.float).view(-1, self.dim_state_action)
        s_ = torch.tensor([t.s_ for t in transitions],
                          dtype=torch.float).view(-1, self.dim_state)
        #actually train cost
        r = -torch.tensor([t.r for t in transitions], dtype=torch.float).view(
            -1, 1)
        s_a, s_, r = s_a.to(device), s_.to(device), r.to(device)

        #get value target
        with torch.no_grad():
            v_target = r + self.gamma * self.target_value(s_)

        # update transition model
        pred_state = self.trans_model(s_a)
        trans_loss = F.mse_loss(pred_state, s_)
        print("transition loss: {}".format(trans_loss.item()))

        self.optimizer_t.zero_grad()
        trans_loss.backward()
        self.optimizer_t.step()

        # update reward model
        pred_reward = self.reward_model(s_a)
        reward_loss = F.mse_loss(pred_reward, r)
        print("reward loss: {}".format(reward_loss.item()))

        self.optimizer_r.zero_grad()
        reward_loss.backward()
        self.optimizer_r.step()

        # update value model
        pred_value = self.value_model(s_)
        value_loss = F.mse_loss(pred_value, v_target)
        print("value loss: {}".format(value_loss.item()))

        self.optimizer_v.zero_grad()
        value_loss.backward()
        self.optimizer_v.step()

        # update value target
        # TODO: change it to EMA?
        if self.training_step % self.target_update == 0:
            self.target_value.load_state_dict(self.value_model.state_dict())
