# MBMF Combination Via Value Expansion #

## 1.Description ##

## 2.Getting started ##
Run the following commands to install this repository and the required dependencies:
```
git clone https://github.com/clthegoat/DL_MBMF.git
pip3 install -r requirements.txt
```
You can run the experiment on the simplest environment 'Pendulum-v1' after installing the required packages, but if you want to run experiments on other more complicated environment, please install mujoco_py [here](https://github.com/openai/mujoco-py). You can run code as follows:
```
cd experiment
python MBMF.py --conf configs/MBMF_Pendulum --type [type name]
```

