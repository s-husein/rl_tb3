import torch.nn as nn
import numpy as np
from gym import Env
from gymenv import Gym

def make_dnn(env: Env, hid_layers = [64, 64], action_space='disc', net_type='shared'):
    layers = []
    inp = np.prod(env.observation_space.shape)
    layers.append(nn.Linear(inp,hid_layers[0]))

    dim_pairs = zip(hid_layers[:-1], hid_layers[1:])
    for in_dim, out_dim in list(dim_pairs):
        layers.append(nn.Linear(in_dim, out_dim))

    if action_space == 'disc':
        out = env.action_space.n
    elif action_space == 'cont':
        out = np.prod(2*len(env.action_space.sample()))
    
    net_types = {'actor': out, 'critic': 1, 'shared': out+1}
    
    layers.append(nn.Linear(hid_layers[-1], net_types[net_type]))

    return nn.Sequential(*layers)

def make_discretize_dnn(env: Env, hid_layers = [128, 128], bins = 7, ordinal=False):
    layers = []
    inp = np.prod(env.observation_space.shape)
    layers.append(nn.Linear(inp,hid_layers[0]))

    dim_pairs = zip(hid_layers[:-1], hid_layers[1:])
    for in_dim, out_dim in list(dim_pairs):
        layers.append(nn.Linear(in_dim, out_dim))

    action_dims = len(env.action_space.sample())

    total_bins = bins*action_dims

    layers.append(nn.Linear(hid_layers[-1], total_bins))

    return nn.Sequential(*layers)


env = Gym(disc_action=False)

actor = make_discretize_dnn(env)


print(actor)


