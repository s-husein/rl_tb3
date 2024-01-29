import torch.nn as nn
import torch
import numpy as np
from gym import Env

class Ordinal(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self._action_dim = action_dim

    def forward(self, x):
        x = torch.sigmoid(x)
        x1, x2 = torch.chunk(x, self._action_dim, dim = -1)
        # y = torch.cat((self.create_ordinal(x1), self.create_ordinal(x2)), dim=-1)
        return x1, x2
    
    def create_ordinal(self, logits):
        dims = logits.dim() - 1 
        repeat_n  = max(min(logits.size()[0]*dims, logits.size()[0]*dims), 1)
        iden = torch.tril(torch.ones_like(torch.eye(logits.size()[-1])))[np.newaxis, :].repeat(repeat_n, 1, 1)
        inv_iden = 1 - iden
        if dims > 0:
            logits = logits[:, np.newaxis]
        return torch.sum(torch.log(iden*(logits) + inv_iden*(1-logits)), dim=-1).squeeze()




def make_dnn(env: Env, hid_layers = [64, 64], action_space='disc', net_type='shared', bins=None, act_fn='relu', ordinal=False):
    layers = []
    activation_fun = {'relu': nn.ReLU(), 'softplus':nn.Softplus(), 'tanh':nn.Tanh()}
    inp = np.prod(env.observation_space.shape)
    layers.append(nn.Linear(inp, hid_layers[0]))
    layers.append(activation_fun[act_fn])
    action_dim = len(env.action_space.sample())

    dim_pairs = zip(hid_layers[:-1], hid_layers[1:])
    for in_dim, out_dim in list(dim_pairs):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(activation_fun[act_fn])

    if action_space == 'disc':
        out = env.action_space.n
    elif action_space == 'cont':
        out = 2*action_dim
    elif action_space == 'discretize':
        out = action_dim*bins
            
    net_types = {'actor': out, 'critic': 1, 'shared': out+1}
    
    layers.append(nn.Linear(hid_layers[-1], net_types[net_type]))
    
    if ordinal:
        layers.append(Ordinal(action_dim))

    return nn.Sequential(*layers)


