import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from .models.model_based import trans_model, reward_model


class Memory():

    data_pointer = 0
    isfull = False

    def __init__(self, capacity):
        self.memory = np.empty(capacity, dtype=object)
        self.capacity = capacity

    def update(self, transition):
        self.memory[self.data_pointer] = transition
        self.data_pointer += 1
        if self.data_pointer == self.capacity:
            self.data_pointer = 0
            self.isfull = True

    def sample(self, batch_size):
        return np.random.choice(self.memory, batch_size)


class Agent():

    def __init__(self, conf):
        self.conf = conf
        self.dim_state = self.conf.data.state.dim
        self.dim_action = self.conf.data.action.dim
        self.dim_state_action = self.dim_state + self.dim_action
        self.trans_model = trans_model(self.dim_state, self.dim_action)
        self.reward_model = reward_model(self.dim_state, self.dim_action)
        self.memory = Memory(self.conf.data.mem_capacity)
        self.optimizer_t = optim.Adam(self.trans_model.parameters(), lr=1e-3)
        self.optimizer_r = optim.Adam(self.reward_model.parameters(), lr=1e-4)

    def select_action(self):
        pass

    def store_transition(self, transition):
        self.memory.update(transition)

    def update(self):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.trans_model.to(device)
        self.reward_model.to(device)

        # data preparation
        transitions = self.memory.sample(self.conf.data.mem_batchsize)
        s_a = torch.tensor([t.s_a for t in transitions], dtype=torch.float).view(-1, self.dim_state_action)
        s_ = torch.tensor([t.s_ for t in transitions], dtype=torch.float).view(-1, self.dim_state)
        r = torch.tensor([t.r for t in transitions], dtype=torch.float).view(-1, 1)
        s_a, s_, r = s_a.to(device), s_.to(device), r.to(device)

        # update transition model
        pred_state = self.trans_model(s_a)
        trans_loss = F.mse_loss(pred_state, s_)

        self.optimizer_t.zero_grad()
        trans_loss.backward()
        self.optimizer_t.step()

        # update reward model
        pred_reward = self.reward_model(s_a)
        reward_loss = F.mse_loss(pred_reward, r)

        self.optimizer_r.zero_grad()
        reward_loss.backward()
        self.optimizer_r.step()