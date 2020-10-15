import argparse
import logging
import torch
import gym
import numpy as np

from collections import namedtuple
from omegaconf import OmegaConf
from Agent import Agent

from Pendulum import PendulumEnv


# basic setting
Transition = namedtuple('Transition', ['s_a', 's_', 'r'])


def main(conf):
    print('****** begin! ******')
    env = PendulumEnv()
    agent = Agent(conf)

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
        if i>200:
            env.state = np.asarray([0.,0.])
            init_state = env._get_obs()
            env.last_u = None
        
        state_list.append(init_state)
        

        episode_reward = 0
        for j in range(trial_len):
            # print('step {} in episode {}'.format(j,i))
            # here should be replace with action solved by LQR
            if i<200:
                action = env.action_space.sample()
            else:
                action = agent.select_action(j, state_list[j], 1)
            state_action = np.concatenate((state_list[j], action))

            # environment iteraction
            #print(env.state)
            gt_state, gt_reward, done, info = env.step(action)
            state_list.append(torch.tensor(gt_state))

            # memory store
            agent.store_transition(Transition(state_action, gt_state, gt_reward))
            episode_reward += gt_reward
            #render
            env.render()
        # train
        if agent.memory.count>256:
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
