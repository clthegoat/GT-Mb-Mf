# MBMF Combination Via Value Expansion #

## 1.Description ##
Reinforcement learning (RL) algorithms have been shown to be ca-pable of learning a wide range of robotic skills. They are divided into two categories: model-based RL and model-free RL. In this project, we propose our method that could gradually transform a model-based RL training framework to a model-free actor-critic ar-chitecture such as Deep Deterministic Policy Gradient (DDPG). The framework is shown as follows.
![alt text](https://github.com/clthegoat/DL_MBMF/blob/main/experiment/assets/framework_reduction.png?raw=true)

## 2.Getting started ##
Run the following commands to install this repository and the required dependencies:
```
git clone https://github.com/clthegoat/DL_MBMF.git
pip3 install -r requirements.txt
```
You can run the experiment on the simplest environment 'Pendulum-v1' after installing the required packages, but if you want to run experiments on other more complicated environment, please install mujoco_py [here](https://github.com/openai/mujoco-py). You can run code as follows:
```
cd experiment
python MBMF.py --conf configuration_mbmf --type [type name]
```
## 3.Experiments ##
Besides running on different environments, you could try different metrics, i.e. *DDPG*, *MVE*, *MPC* to compare with our method *MBMF* via modifying parameters in configuration file *configuration_mbmf*.
*MBMF*
```
train.Agent_Type: MBMF
MVE.horizon: 5
```
*DDPG*
```
train.Agent_Type: MBMF
MVE.horizon: 0
```
*MPC*
```
train.Agent_Type: MPC
MVE.horizon: 5
```
*MVE*
```
train.Agent_Type: MVE
MVE.horizon: 5
```
