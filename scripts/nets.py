import torch.nn as nn
import torch
import numpy as np
from gym import Env
import gym

class Ordinal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.sigmoid(x)
        dims = x.dim()
        iden = torch.tril(torch.ones_like(torch.eye(x.size()[-1])))[np.newaxis, :, :].repeat(x.size()[0], 1, 1)
        inv_iden = 1 - iden
        x = x[:, np.newaxis]
        return torch.sum(torch.log(iden*(x) + inv_iden*(1-x)), dim=2).squeeze()



def make_dnn(env: Env, hid_layers = [64, 64], action_space='disc', net_type='shared', bins=None, act_fn='relu', ordinal=False):
    layers = []
    activation_fun = {'relu': nn.ReLU(), 'softplus':nn.Softplus(), 'tanh':nn.Tanh()}
    inp = np.prod(env.observation_space.shape)
    layers.append(nn.Linear(inp, hid_layers[0]))
    layers.append(activation_fun[act_fn])

    dim_pairs = zip(hid_layers[:-1], hid_layers[1:])
    for in_dim, out_dim in list(dim_pairs):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(activation_fun[act_fn])

    if action_space == 'disc':
        out = env.action_space.n
    elif action_space == 'cont':
        out = np.prod(2*len(env.action_space.sample()))
    elif action_space == 'discretize':
        out = len(env.action_space.sample())*bins
            
    net_types = {'actor': out, 'critic': 1, 'shared': out+1}
    
    layers.append(nn.Linear(hid_layers[-1], net_types[net_type]))
    
    if ordinal:
        layers.append(Ordinal())

    return nn.Sequential(*layers)


