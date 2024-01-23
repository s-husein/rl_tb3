import torch.nn as nn
import torch
import numpy as np
from gym import Env
import gym

class Ordinal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.clip(min=1e-6)
        iden = torch.tril(torch.ones(2*x.size()))
        inv_iden = 1-iden
        return torch.sum(iden*torch.log(x)+inv_iden*torch.log(1-x), dim=1)


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

def make_discretize_dnn(env: Env, hid_layers = [128, 128], act_fn='relu', bins = 7, ordinal=False):
    layers = []
    activation_fun = {'relu': nn.ReLU(), 'softplus':nn.Softplus(), 'tanh':nn.Tanh()}

    inp = np.prod(env.observation_space.shape)
    layers.append(nn.Linear(inp,hid_layers[0]))
    layers.append(activation_fun[act_fn])

    dim_pairs = zip(hid_layers[:-1], hid_layers[1:])
    for in_dim, out_dim in list(dim_pairs):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(activation_fun[act_fn])

    action_dims = len(env.action_space.sample())

    total_bins = bins*action_dims

    layers.append(nn.Linear(hid_layers[-1], total_bins))

    if ordinal:
        layers.append(Ordinal())

    return nn.Sequential(*layers)

env = gym.make('LunarLanderContinuous-v2')


state = torch.tensor(env.reset()[0])
# states = []

# for i in range(10):
#     states.append(torch.tensor(env.reset()[0]))

# states = torch.stack(states)

# print(f'states: {states}')


actor = make_discretize_dnn(env, bins = 3, ordinal=True, act_fn='relu')

print(actor)

print(actor(state))

# print(actor(states))


