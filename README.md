# Gradual Transition From Model-Based to Model-Free Actor-Critic Reinforcement Learning (GT-Mb-Mf) #
This repo contains the code for our Deep Learning course project. 
 <br> Developed by Le Chen, Yunke Ao, Kaiyue Shen and Zheyu Ye.

## 1.Description ##
Reinforcement learning (RL) algorithms have been shown to be capable of learning a wide range of robotic skills. Model-free RL algorithms directly optimize the policy based on gathered interaction experiences, while model-based ones additionally learn the dynamic and reward functions of the environment. Generally, model-free RL could achieve higher performance for different tasks but typically with millions of trials for convergence. For model-based RL, much fewer samples are required for learning a decent policy, but the existence of bias in learned model usually results in relatively lower final performance compared with model-free RL algorithms. 
In this project, our approach starts with a model-based framework which is integrated with value learning, and gradually transform it to pure model-free actor-critic learning by reducing the planning horizon. The whole framework is shown as follows.
![alt text](https://github.com/clthegoat/GT-Mb-Mf/blob/main/experiment/assets/framework_reduction.png?raw=true)

## 2.Getting started ##
Run the following commands to install this repository and the required dependencies:
```
git clone https://github.com/clthegoat/GT-Mb-Mf.git
pip3 install -r requirements.txt
```
You can run the experiment on the simplest environment 'Pendulum-v1' after installing the required packages, but if you want to run experiments on other more complicated environment, please install mujoco_py [here](https://github.com/openai/mujoco-py). You can run code as follows:
```
cd experiment
python GTMBMF.py --conf configs/configuration_gtmbmf --type [type name]
```
The type names include 'pendulum', 'luna', 'halfcheet', 'walker' and 'hopper', corresponding to 'Pendulum-v1', 'LunarLanderContinuous-v2', 'HalfCheetah-v2', 'Walker2d-v2' and 'Hopper-v2' gym environments respectively.

## 3.Experiments ##
Besides running on different environments, you could try different baselines, i.e. *DDPG*, *MVE*, *MPC* to compare with our method *GT-Mb-Mf* via modifying parameters in configuration file *experiment/configs/configuration_gtmbmf*.

*GT-Mb-Mf*
```
train.Agent_Type: GT-Mb-Mf
planning.horizon: 5
```
*DDPG*
```
train.Agent_Type: GT-Mb-Mf
planning.horizon: 0
```
*MPC*
```
train.Agent_Type: MPC
planning.horizon: 5
```
*MVE*
```
train.Agent_Type: MVE
MVE.horizon: 2
```
