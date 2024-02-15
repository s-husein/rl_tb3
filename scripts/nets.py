import torch.nn as nn
import torch
import numpy as np
from gym import Env
from gymenv import Gym

class Ordinal(nn.Module):
    def __init__(self, action_dim, ordinal):
        super().__init__()
        self._action_dim = action_dim
        self._ordinal = ordinal

    def forward(self, x):
        probs_ = torch.chunk(x, self._action_dim, dim = -1)
        if self._ordinal:
            return [torch.softmax(self.create_ordinal(torch.sigmoid(prob_)), dim=-1) for prob_ in probs_]
        else:
            return [torch.softmax(prob_, dim=-1) for prob_ in probs_]
    
    def create_ordinal(self, logits):
        dims = logits.dim() - 1 
        repeat_n  = max(min(logits.size()[0]*dims, logits.size()[0]*dims), 1)
        iden = torch.tril(torch.ones_like(torch.eye(logits.size()[-1])))[np.newaxis, :].repeat(repeat_n, 1, 1)
        inv_iden = 1 - iden
        if dims > 0:
            logits = logits[:, np.newaxis]
        return torch.sum(torch.log(iden*(logits) + inv_iden*(1-logits)), dim=-1).squeeze()


def make_dnn(env: Env, hid_layers = [64, 64], action_space='disc', net_type='shared', bins=None,
             act_fn='relu', ordinal=False, conv_layers=None, max_pool = None):
    
    layers = []
    activation_fun = {'relu': nn.ReLU(), 'softplus':nn.Softplus(), 'tanh':nn.Tanh(), 'elu': nn.ELU()}
    inp_shape = env.observation_space.shape

    if conv_layers is not None:
        in_chann = 1
        inp_h, inp_w = inp_shape[0], inp_shape[1]
        for conv in conv_layers:
            out_chann, filter_size, stride = conv
            layers.append(nn.Conv2d(in_chann, out_chann, filter_size, stride))
            layers.append(nn.ELU())

            out_h = (inp_h - filter_size)//stride + 1
            out_w = (inp_w - filter_size)//stride + 1
            inp_h = out_h
            inp_w = out_w

            if max_pool is not None:
                layers.append(nn.MaxPool2d(max_pool[0], max_pool[1]))
                out_h = (inp_h - max_pool[0])//max_pool[1] + 1
                out_w = (inp_w - max_pool[0])//max_pool[1] + 1
                inp_h = out_h
                inp_w = out_w
            in_chann = out_chann

        layers.append(nn.Flatten())
        layers.append(nn.Linear(inp_h*inp_w*in_chann, hid_layers[0]))
        layers.append(activation_fun[act_fn])

    else:
        inp = np.prod(inp_shape)
        layers.append(nn.Linear(inp, hid_layers[0]))
        layers.append(activation_fun[act_fn])

    try:
        action_dim = len(env.action_space.sample())
    except:
        pass
    
    if len(hid_layers) > 1:
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
    if net_type in net_types.keys():
        
        layers.append(nn.Linear(hid_layers[-1], net_types[net_type]))
        
        if action_space == 'discretize' and net_type == 'actor':
            layers.append(Ordinal(action_dim, ordinal))

    return nn.Sequential(*layers)



env = Gym(obs_scale_factor=0.1)

conv_l= [[16, 3, 1],
         [32, 3, 1],
         [64, 3, 1]]

actor = make_dnn(env, hid_layers = [3], conv_layers=conv_l, max_pool=[2,2], net_type='rnd', act_fn='elu').to('cuda')
print(actor)
states = []
for i in range(5):
    state = torch.tensor(env.reset()[0]).to('cuda')
    states.append(state)

states_ = torch.stack(states)

print(actor(state))