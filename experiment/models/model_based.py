import torch
import torch.nn as nn
import numpy as np


class trans_model(nn.Module):
    def __init__(self, dim_state, dim_action):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_state + dim_action, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, dim_state)
        )

    def forward(self, x):
        return self.fc(x)


class reward_model(nn.Module):
    def __init__(self, dim_state, dim_action):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_state + dim_action, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)

class value_model(nn.Module):
    def __init__(self, dim_state):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_state, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):


        return self.fc(x)

class actor_model(nn.Module):
    def __init__(self, dim_state, dim_action):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_state, 100),
            nn.ReLU(),
            nn.Linear(100, dim_action)
        )

    def forward(self, x):
        return self.fc(x)


class critic_model(nn.Module):
    def __init__(self, dim_state, dim_action):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_state + dim_action, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):

        return self.fc(x)
