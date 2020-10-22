import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from models.model_based import actor_model, critic_model, trans_model, reward_model
# from Pendulum import PendulumEnv
from MPC_agent import Memory
import copy
import numpy as np
import random


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
        dx = self.theta * (self.mu - self.state) + self.sigma * np.array([np.random.normal() for _ in range(len(self.state))])
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

class MVE_agent():
    """ A MVE Agent """

    def __init__(self, conf):
        self.conf = conf
        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        """ model parameters"""
        self.num_random = self.conf.train.num_random # number of first random trails
        self.tau = self.conf.MVE.target_model_update_rate
        self.gamma = self.conf.train.gamma # reward discount
        self.T = self.conf.MVE.horizon     # forward predict steps
        self.trail_len = self.conf.trail_len # steps in each trail
        self.batch_size = self.conf.data.mem_batchsize
        self.iter_num = self.conf.MVE.iteration_num # iteration num after selecting an action
        self.dim_state = self.conf.data.state.dim
        self.dim_action = self.conf.data.action.dim
        self.dim_state_action = self.dim_state + self.dim_action
        # for "pendulumEnv", state & action limitation
        self.up_X = torch.Tensor([1.,1.,8.]).expand(self.batch_size,-1)
        self.low_X = -self.up_X
        self.up_U = 2.
        self.low_U = -self.up_U
        """ models"""
        self.trans_model = trans_model(self.dim_state, self.dim_action).to(self.device)
        self.reward_model = reward_model(self.dim_state, self.dim_action).to(self.device)
        self.critic_local = critic_model(self.dim_state, self.dim_action).to(self.device)
        self.critic_target = critic_model(self.dim_state, self.dim_action).to(self.device)
        self.copy_model_over(self.critic_local, self.critic_target)
        self.actor_local = actor_model(self.dim_state, self.dim_action).to(self.device)
        self.actor_target = actor_model(self.dim_state, self.dim_action).to(self.device)
        self.copy_model_over(self.actor_local, self.actor_target)
        self.memory = Memory(self.conf.data.mem_capacity)
        self.optimizer_t = optim.Adam(self.trans_model.parameters(), lr=1e-3)
        self.optimizer_r = optim.Adam(self.reward_model.parameters(), lr=1e-3)
        self.optimizer_c = optim.Adam(self.critic_local.parameters(), lr=1e-3)
        self.optimizer_a = optim.Adam(self.actor_local.parameters(), lr=1e-4)
        self.exploration_strategy = OU_Noise_Exploration(self.dim_action)
        self.training_step = 0
        # self.select_action()
        # self.store_transition()
        # self.sample_transitions()
        self.max_grad_norm = 0.01
        

    def select_action(self,state):
        """Picks an action using the actor network and then adds some noise to it to ensure exploration"""
        state = state.unsqueeze(0).to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action = self.exploration_strategy.perturb_action_for_exploration_purposes(action)
        action = np.clip(action, self.low_U, self.up_U)
        return action.squeeze(0)

    def store_transition(self, transition):
        self.memory.update(transition)

    def sample_transitions(self, num_trans=None):
        """random sample certain number (default: batchsize) of transitions from Memory"""
        if num_trans is not None:
            batch_size = num_trans
        else:
            batch_size = self.batch_size
        
        transitions = self.memory.sample(batch_size)
        s = torch.from_numpy(np.vstack((t.s for t in transitions if t is not None))).float().to(self.device)
        # s = torch.tensor([t.s for t in transitions], dtype=torch.float).view(-1, self.dim_state).to(self.device)
        a = torch.tensor([t.a for t in transitions], dtype=torch.float).view(-1, self.dim_action).to(self.device)
        s_a = torch.tensor([t.s_a for t in transitions], dtype=torch.float).view(-1, self.dim_state_action).to(self.device)
        s_ = torch.tensor([t.s_ for t in transitions], dtype=torch.float).view(-1, self.dim_state).to(self.device)
        r = torch.tensor([t.r for t in transitions], dtype=torch.float).view(-1, 1).to(self.device)
        return s, a, s_a, r, s_, #dones

    def update(self):
        """ update transition, reward, actor, critic model"""
        self.training_step += 1
        for _ in range(self.iter_num):
            # s, a, s_a, r, s_, dones = self.sample_transitions()
            s, a, s_a, r, s_ = self.sample_transitions()
            trans_loss = self.trans_learn(s_a, s_)
            reward_loss = self.reward_learn(s_a, r)
            # env = PendulumEnv() # used to get true reward
            actor_loss = self.actor_learn(s)
            critic_loss = self.critic_learn(s_a, r, s_)
        if self.training_step % 20:
            print("transition loss: {}".format(trans_loss))
            print("reward loss: {}".format(reward_loss))
            print("actor loss: {}".format(actor_loss))
            print("critic loss: {}".format(critic_loss))
        

    def trans_learn(self, states_actions, next_states):
        """Runs a learning iteration for the transition model"""
        # states_pred = self.trans_model(torch.cat((states, actions), 1))
        states_pred = self.trans_model(states_actions)
        trans_loss = F.mse_loss(states_pred, next_states)
        self.optimizer_t.zero_grad()
        trans_loss.backward()
        self.optimizer_t.step()

        return trans_loss.item()

    def reward_learn(self, states_actions, rewards):
        """ update reward model"""
        reward_pred = self.reward_model(states_actions)
        reward_loss = F.mse_loss(reward_pred, rewards)
        self.optimizer_r.zero_grad()
        reward_loss.backward()
        self.optimizer_r.step()

        return reward_loss.item()

    def actor_learn(self, states):
        """Runs a learning iteration for the actor"""
        # if self.done: #we only update the learning rate at end of each episode
        #     self.update_learning_rate(self.hyperparameters["Actor"]["learning_rate"], self.actor_optimizer)
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(torch.cat((states, actions_pred), 1)).mean()
        self.optimizer_a.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_local.parameters(), self.max_grad_norm)
        self.optimizer_a.step()
        self.soft_update_of_target_network(self.actor_local, self.actor_target, self.tau)

        return actor_loss.item()

    def critic_learn(self, states_actions, rewards, next_states):
        """Runs a learning iteration for the critic"""
        imag_rewards_list = [rewards]
        # compute Q(s^t, a^t) for t~[-1,T]
        critic_pred = torch.empty(self.batch_size, self.T+1, dtype=torch.float)
        critic_pred[:,0:1] = self.critic_local(states_actions)

        for t in range(1,self.T+1): # model-based: predict T steps forward
            states = next_states
            with torch.no_grad():
                actions = self.actor_target(states)
                actions = torch.clamp(actions, self.low_U, self.up_U)
            states_actions = torch.cat((states, actions), 1)
            critic_pred[:,t:t+1] = self.critic_local(states_actions)
            next_states = self.trans_model(states_actions)
            next_states = torch.max(torch.min(next_states,self.up_X),self.low_X)
            rewards = self.reward_model(states_actions)
            # env.state = states
            # _, rewards, _, _ = env.step(actions)
            imag_rewards_list.append(rewards)

        critic_pred = critic_pred[:,:-1] # Q(s^t,a^t), t~[-1,T-1]
        final_states = states # s_T
        with torch.no_grad():
            final_actions = self.actor_target(final_states) # a_T
            final_critic = self.critic_target(torch.cat((final_states, final_actions), 1)) # Q'(s_T,a_T)
            # compute sum of discounted reward and ternimal cost for each timestep t~[-1,T-1]
            critic_target = torch.empty(self.batch_size, self.T, dtype=torch.float)
            critic_target[:,self.T-1:self.T] = imag_rewards_list[self.T-1] + self.gamma * final_critic
            #add -1
            for t in range(self.T-1,-1,-1):
                critic_target[:,t-1:t] = imag_rewards_list[t-1] + self.gamma * critic_target[:,t:t+1]
        #print(critic_pred.size())
        critic_loss = F.mse_loss(critic_pred, critic_target) / (self.T)
        self.optimizer_c.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.max_grad_norm)
        self.optimizer_c.step()
        self.soft_update_of_target_network(self.critic_local, self.critic_target, self.tau)

        return critic_loss.item()
   
    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    @staticmethod
    def copy_model_over(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())
