from nets import make_net
import torch
import cv2 as cv
import numpy as np
from algos import PPO
# from gymenv import Gym
import gymnasium as gym
from yaml import safe_load


pu = 'cuda' if torch.cuda.is_available() else 'cpu'


with open('../config.yaml') as file:
    params = safe_load(file)

env = params['env'] = gym.make("BipedalWalker-v3", render_mode="human")

make_net(params)

agent = PPO(**params['algo_params'], **params['network_params'])

print(agent.configs)

