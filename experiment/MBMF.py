import argparse
import logging
import torch
import gym
import numpy as np

from collections import namedtuple
from omegaconf import OmegaConf
from .Agent import Agent


# basic setting
Transition = namedtuple('Transition', ['s_a', 's_', 'r'])


def main(conf):

    env = gym.make(conf.data.name)
    agent = Agent(conf)

    # train setting
    num_trials = conf.train.num_trials
    trial_len = conf.train.trail_len

    for i in range(num_trials):

        # initial state
        state_list = []
        init_state = env.reset()
        state_list.append(init_state)

        for j in range(trial_len):

            # here should be replace with action solved by LQR
            action = env.action_space.sample()
            state_action = np.concatenate((state_list[j], action))

            # environment iteraction
            gt_state, gt_reward, done, info = env.step(action)
            state_list.append(torch.tensor(gt_state))

            # memory store
            agent.store_transition(Transition(state_action, gt_state, gt_reward))

            # train
            if agent.memory.isfull:
                agent.update()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str)
    args = parser.parse_args()

    conf = OmegaConf.load(args.conf)

    main(conf)