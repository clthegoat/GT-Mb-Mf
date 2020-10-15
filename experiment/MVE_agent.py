import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from .models.model_based import actor_model, critic_model, trans_model
from pendulum import PendulumRnv
from Agent import Memory
import copy
import numpy as np

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

    def __init__(self, config):
        self.config = config
        self.tau = self.config.train.MVE.tau
        self.discount_rate = self.config.train.MVE.discount_rate
        self.H_step = self.config.train.MVE.H_step
        # self.num_iter = self.config.train.MVE.actor_critic_iter_perstep  #put this line in MBMF.py
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dim_state = self.config.data.state.dim
        self.dim_action = self.config.data.action.dim
        self.trans_model = trans_model(self.dim_state, self.dim_action)
        self.critic_local = critic_model(self.dim_state, self.dim_action).to(self.device)
        self.critic_target = critic_model(self.dim_state, self.dim_action).to(self.device)
        copy_model_over(self.critic_local, self.critic_target)
        self.actor_local = actor_model(self.dim_state, self.dim_action).to(self.device)
        self.actor_target = actor_model(self.dim_state, self.dim_action).to(self.device)
        copy_model_over(self.actor_local, self.actor_target)
        self.memory = Memory(self.config.data.mem_capacity)
        self.optimizer_t = optim.Adam(self.trans_model.parameters(), lr=1e-3)
        self.optimizer_c = optim.Adam(self.critic_local.parameters(), lr=1e-3)
        self.optimizer_a = optim.Adam(self.actor_local.parameters(), lr=3e-4)
        self.exploration_strategy = OU_Noise_Exploration(self.dim_action)

    def select_action(self,state):
        """Picks an action using the actor network and then adds some noise to it to ensure exploration"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action = self.exploration_strategy.perturb_action_for_exploration_purposes(action)
        return action.squeeze(0)

    def store_transition(self, transition):
        self.memory.update(transition)

    def sample_transitions(self, num_trans=None):
        """random sample certain number (default: batchsize) of transitions from Memory"""
        if num_trans is not None:
            batch_size = num_trans
        else:
            batch_size = self.config.data.mem_batchsize
        transitions = self.memory.sample(batch_size)
        states = torch.from_numpy(np.vstack([t.state for t in transitions if t is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([t.action for t in transitions if t is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([t.reward for t in transitions if t is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([t.next_state for t in transitions if t is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([int(t.done) for t in transitions if t is not None])).float().to(self.device)
        return states, actions, rewards, next_states, dones

    def update(self, num_iter=10):
        """ update transition model"""
        states, actions, _, next_states, _ = sample_transitions()
        trans_learn(states, actions, next_states) # now update once for each step, can also be put in the following loop to update several times
        
        """ update actor & critic model"""
        env = PendulumRnv() # used to get true reward
        for _ in range(num_iter):
            states, actions, rewards, next_states, dones = sample_transitions()
            actor_learn(states)
            critic_learn(self, states, actions, rewards, next_states, dones, env)

    def trans_learn(self, states, actions, next_states):
        """Runs a learning iteration for the transition model"""
        states_pred = self.trans_model(torch.cat((states, actions), 1))
        trans_loss = F.mse_loss(states_pred, next_states)
        self.optimizer_t.zero_grad()
        trans_loss.backward()
        self.optimizer_t.step()

    def actor_learn(self, states):
        """Runs a learning iteration for the actor"""
        # if self.done: #we only update the learning rate at end of each episode
        #     self.update_learning_rate(self.hyperparameters["Actor"]["learning_rate"], self.actor_optimizer)
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(torch.cat((states, actions_pred), 1)).mean()
        self.optimizer_a.zero_grad()
        actor_loss.backward()
        self.optimizer_a.step()
        self.soft_update_of_target_network(self.actor_local, self.actor_target, self.tau)

    def critic_learn(self, states, actions, rewards, next_states, dones, env):
        """Runs a learning iteration for the critic"""
        imag_rewards_list = [rewards]
        critic_pred = torch.empty(states.size(0), H_step+1, dtype=torch.float)
        critic_pred[:,0] = self.critic_local(torch.cat((states, actions), 1))
        for t in range(H_step): # model-based: predict H steps forward
            states = next_states
            actions = self.actor_target(states)
            critic_pred[:,t+1] = self.critic_local(torch.cat((states, actions), 1))
            next_states = self.trans_model(torch.cat((states, actions), 1))
            env.state = states
            _, rewards, _, _ = env.step(actions)
            imag_rewards_list.append(rewards)
        critic_pred = critic_pred[:,:-1] # Q(s_t,a_t), t~[-1,H-1]
        final_states = next_states # s_H
        final_actions = self.actor_target(final_states) # a_H
        final_critic = self.critic_target(torch.cat((final_states, final_actions), 1)) # Q'(s_H,a_H)

        critic_target = torch.empty(states.size(0), H_step, dtype=torch.float) # MVE
        critic_target[:,H_step-1] = imag_rewards_list[-1] + self.discount_rate * final_critic
        for t in range(1,H_step):
            critic_target[:,H_step-1-t] = imag_rewards_list[H_step-1-t] + self.discount_rate * critic_target[:,H_step-t]
        critic_loss = F.mse_loss(critic_pred, critic_target) / H_step
        self.optimizer_c.zero_grad()
        critic_loss.backward()
        self.optimizer_c.step()
        self.soft_update_of_target_network(self.critic_local, self.critic_target, self.tau)
   
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
