import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.cuda as cuda
import numpy as np
from gym import Env
from gymenv import Gym
import gym 


device = 'cuda' if cuda.is_available() else 'cpu'
print(f'using {device}')


def make_dnn(env: Env, hid_layers = [64, 64], action_space='disc', net_type='shared'):
    layers = []
    inp = np.prod(env.observation_space.shape)
    layers.append(nn.Linear(inp,hid_layers[0]))

    dim_pairs = zip(hid_layers[:-1], hid_layers[1:])
    for in_dim, out_dim in list(dim_pairs):
        layers.append(nn.Linear(in_dim, out_dim))

    if action_space == 'disc':
        out = env.action_space.n
    
    net_types = {'actor': out, 'critic': 1, 'shared': out+1}
    
    layers.append(nn.Linear(hid_layers[-1], net_types[net_type]))

    return nn.Sequential(*layers)


env = gym.make('CartPole-v1')

print(make_dnn(env, net_type='actor'))




