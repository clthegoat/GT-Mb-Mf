import argparse
# from experiment.Agent import Agent
import logging
import torch
import gym
import numpy as np

from collections import namedtuple
from omegaconf import OmegaConf
from MPC_agent import *
from MVE_agent import *
from MBMF_agent import *
from Pendulum import PendulumEnv


# basic setting
# Transition = namedtuple('Transition', ['s_a', 's_', 'r'])
# Ext_transition = namedtuple('Ext_transition', ['state', 'action', 'state_action', 'next state', 'reward', 'done'])
# Ext_transition = namedtuple('Ext_transition', ['s', 'a', 's_a', 's_', 'r', 'done'])
# MBMF_transition = namedtuple('MBMF_transition', ['s', 'a', 's_a', 's_', 'r', 't', 'done'])
Ext_transition = namedtuple('MBMF_transition', ['s', 'a', 's_a', 's_', 'r', 't', 'done'])

def main(conf):
    print('****** begin! ******')
    env = PendulumEnv()
    Agent_Type = conf.train.Agent_Type
    if Agent_Type == "MPC":
        agent = MPC_agent(conf)
    elif Agent_Type == "MVE":
        agent = MVE_agent(conf)
    elif Agent_Type == "MBMF":
        agent = MBMF_agent(conf)
    else:
        raise ValueError
    print('****** step1 ******')
    # train setting
    num_trials = conf.train.num_trials
    trial_len = conf.train.trail_len

    for i in range(num_trials):
        # print('episode {}'.format(i))
        # initial state
        state_list = []
        #init_state = env.reset()
        #reset the state to be a single start point:
        
        init_state = env.reset()
        # if i>agent.num_random:
        #     env.state = np.asarray([0.,0.])
        #     init_state = env._get_obs()
        #     env.last_u = None
        #     print(init_state)
            #print(agent.value_model(torch.from_numpy(init_state).float()))
        state_list.append(torch.tensor(init_state, dtype=torch.float))
        

        episode_reward = 0
        for j in range(trial_len):
            # print('step {} in episode {}'.format(j,i))
            # here should be replace with action solved by LQR
            if i<=agent.num_random:
                action = env.action_space.sample()
            else:
                if Agent_Type == "MPC":
                    action = agent.select_action(j, state_list[j], mode=2, exploration=0)
                elif Agent_Type == "MVE":
                    action = agent.select_action(state_list[j])
                
            state_action = np.concatenate((state_list[j], action))

            # environment iteraction
            #print(env.state)
            gt_state, gt_reward, done, info = env.step(action)
            state_list.append(torch.tensor(gt_state, dtype=torch.float))

            # memory store
            # Ext_transition = namedtuple('MBMF_transition', ['s', 'a', 's_a', 's_', 'r', 't', 'done'])
            agent.store_transition(Ext_transition(state_list[j], action, state_action, gt_state, gt_reward, j, done))

            episode_reward += gt_reward
            #render
            if i > agent.num_random and i % 200 == 0:
                env.render()

        # train
        if agent.memory.count>agent.batch_size:
            agent.update()

        #see the trend of reward
        print('episode {}, total reward {}'.format(i,episode_reward)) 


    print('****** done! ******')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str)
    args = parser.parse_args()

    conf = OmegaConf.load(args.conf)

    main(conf)
