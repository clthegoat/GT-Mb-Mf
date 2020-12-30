import argparse
# from experiment.Agent import Agent
import logging
from numpy.lib.function_base import average
import torch
import gym
import numpy as np

from collections import namedtuple
from omegaconf import OmegaConf
from MPC_agent import *
from MVE_agent import *
from MBMF_agent import *
from Pendulum import PendulumEnv

import wandb
wandb.init(project="dl_mbmf")
wandb.config["more"] = "custom"

# basic setting
# Transition = namedtuple('Transition', ['s_a', 's_', 'r'])
# Ext_transition = namedtuple('Ext_transition', ['state', 'action', 'state_action', 'next state', 'reward', 'done'])
# Ext_transition = namedtuple('Ext_transition', ['s', 'a', 's_a', 's_', 'r', 'done'])
# MBMF_transition = namedtuple('MBMF_transition', ['s', 'a', 's_a', 's_', 'r', 't', 'done'])
Ext_transition = namedtuple('MBMF_transition',
                            ['s', 'a', 's_a', 's_', 'r', 't', 'done'])



class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def _reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action


def main(conf, type):
    print('****** begin! ******')
    # env = PendulumEnv()
    if type == 'pendulum':
        env = NormalizedActions(PendulumEnv())
    elif type == 'ant':
        env = NormalizedActions(gym.make('Ant-v2'))
    elif type == 'halfcheet':
        env = NormalizedActions(gym.make('HalfCheetah-v1'))
    elif type == 'walker':
        env = NormalizedActions(gym.make('Walker2d-v2'))
    else:
        env = NormalizedActions(gym.make('Hopper-v2'))


    ##normalize environment

    state_dim = len(env.reset())
    state_high = list(map(float, list(env.observation_space.high)))
    state_low = list(map(float, list(env.observation_space.low)))
    action_dim = len(env.action_space.sample())
    action_high = list(map(float, list(env.action_space.high)))
    action_low = list(map(float, list(env.action_space.low)))

    new_conf = {
        'data': {
            'state': {
                'dim': state_dim,
                'high': state_high,
                'low': state_low,
            },
            'action': {
                'dim': action_dim,
                'high': action_high,
                'low': action_low,
            },
        }
    }
    conf = OmegaConf.merge(conf, new_conf)
    Agent_Type = conf.train.Agent_Type

    # parser = argparse.ArgumentParser()

    # train params
    args.train_num_trials = conf.train.num_trials
    args.train_trail_len = conf.train.trail_len
    args.train_num_random = conf.train.num_random
    args.train_action_noise = conf.train.action_noise
    args.train_gamma = conf.train.gamma
    args.train_target_update_num = conf.train.target_update_num
    args.train_agent_Type = conf.train.Agent_Type
    args.train_K = conf.train.K
    args.train_c1 = conf.train.c1
    args.train_c2 = conf.train.c2

    # data params
    # args.data_name = conf.data.name
    args.data_type = type
    args.data_state_dim = conf.data.state.dim
    args.data_action_dim = conf.data.action.dim
    args.data_mem_capacity = conf.data.mem_capacity
    args.data_mem_batchsize = conf.data.mem_batchsize
    args.data_mb_mem_batchsize = conf.data.mb_mem_batchsize

    # planning params
    args.planning_horizon = conf.planning.horizon
    args.planning_ilqr_learning_rate = conf.planning.ilqr_learning_rate
    args.planning_ilqr_iteration_num = conf.planning.ilqr_iteration_num
    args.planning_shooting_num = conf.planning.shooting_num

    # MVE params
    args.mve_horizon = conf.MVE.horizon
    args.mve_iteration_num = conf.MVE.iteration_num
    args.mve_target_model_update_rate = conf.MVE.target_model_update_rate
    # args = parser.parse_args()

    wandb.config.update(args)  # adds all of the arguments as config variables

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
            if i <= agent.num_random:
                action = env.action_space.sample()
            else:
                if Agent_Type == "MBMF":
                    action = agent.mbmf_select_action(j, state_list[j], exploration=1, relative_step=1).reshape((-1,))
                else:
                    action = agent.select_action(state_list[j], exploration=1)

            
            # print(action.shape)
            state_action = np.concatenate((state_list[j], action))

            # environment iteraction
            #print(env.state)
            gt_state, gt_reward, done, info = env.step(action)
            state_list.append(torch.tensor(gt_state, dtype=torch.float))

            # memory store
            agent.store_transition(
                Ext_transition(state_list[j], action, state_action, gt_state,
                               gt_reward, j, done))

            episode_reward += gt_reward
            #render
            
            #print the automatic deduction process
       
            # train
            if agent.memory.count > agent.batch_size:
                if Agent_Type == "MBMF":
                    if i <= agent.num_random:
                        trans_loss, reward_loss, mb_actor_loss, mb_critic_loss, mf_actor_loss, mf_critic_loss = agent.update(
                            0)
                    else:
                        trans_loss, reward_loss, mb_actor_loss, mb_critic_loss, mf_actor_loss, mf_critic_loss = agent.update(
                            1)
                if Agent_Type == "MVE":
                    trans_loss, reward_loss, mb_actor_loss, mb_critic_loss = agent.update(
                    )

                if Agent_Type == "MPC":
                    if i <= agent.num_random or i>=agent.num_random+agent.fixed_num_per_redction:
                        trans_loss, reward_loss, mb_actor_loss, mb_critic_loss = agent.update(
                        0)
                    else:
                        # if i % agent.num_random==0 and agent.T>1:
                        #     agent.T -= 1
                        trans_loss, reward_loss, mb_actor_loss, mb_critic_loss = agent.update(
                        1)

                #see the trend of reward
                # print('episode {}, total reward {}'.format(i,episode_reward))
                wandb.log({
                    "trans_loss": trans_loss,
                    "reward_loss": reward_loss,
                    "mb_actor_loss": mb_actor_loss,
                    "mb_critic_loss": mb_critic_loss
                })
                if Agent_Type == "MBMF":
                    wandb.log({
                        "mf_actor_loss": mf_actor_loss,
                        "mf_critic_loss": mf_critic_loss
                   })

        wandb.log({
                "episode": i,
                "total reward": episode_reward})

        if i > agent.num_random and Agent_Type == "MBMF":
            if agent.backward:
                print("automotic reduction stage {}".format(agent.K))
            else:
                print("automotic reduction stage {}".format(agent.T))

        #test every 20 episodes
        if i % 10 == 0 and i > 0:
            test_reward_sum = 0
            # print('start test!')
            # test
            for num in range(10):
                # print('test time {}'.format(num))
                test_state_list = []
                # init_state = env.reset()
                # reset the state to be a single start point:
                test_init_state = env.reset()
                test_state_list.append(
                    torch.tensor(test_init_state, dtype=torch.float))
                for step_num in range(trial_len):
                    if num==0:
                        env.render()
                    # print('step {} in episode {}'.format(step_num,i))
                    if Agent_Type == "MVE" or "MPC":
                        test_action = agent.select_action(
                            test_state_list[step_num], exploration=0).reshape((-1,))
                    if Agent_Type == "MBMF":
                        test_action = agent.mbmf_select_action(
                            step_num,
                            test_state_list[step_num],
                            exploration=0,
                            relative_step=1).reshape((-1,))
                    
                    test_state_action = np.concatenate(
                        (test_state_list[step_num], test_action))

                    # environment iteraction
                    # print(env.state)
                    test_gt_state, test_gt_reward, done, info = env.step(
                        test_action)
                    test_state_list.append(
                        torch.tensor(test_gt_state, dtype=torch.float))

                    # memory store
                    agent.store_transition(
                        Ext_transition(test_state_list[step_num], test_action,
                                       test_state_action, test_gt_state,
                                       test_gt_reward, step_num, done))

                    test_reward_sum += test_gt_reward
            average_test_reward_sum = test_reward_sum / 10
            # print(
            #     'average_test_reward_sum = {}'.format(average_test_reward_sum))
            wandb.log({
                "episode": i,
                "average_test_reward_sum": average_test_reward_sum
            })

    print('****** done! ******')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', nargs='?', default='pendulum', help='you need to input the type of experiments including: pendulum(default), ant, halfcheet, walker')
    parser.add_argument('--conf', type=str)
    args = parser.parse_args()

    conf = OmegaConf.load(args.conf)

    type = args.type
    main(conf, type)
