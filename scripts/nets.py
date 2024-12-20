import torch.nn as nn
import torch
import numpy as np
from gym import Env

pu = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        iden = torch.tril(torch.ones_like(torch.eye(logits.size()[-1]).to(pu)))[np.newaxis, :].repeat(repeat_n, 1, 1)
        inv_iden = 1 - iden
        if dims > 0:
            logits = logits[:, np.newaxis]
        return torch.sum(torch.log(iden*(logits) + inv_iden*(1-logits)), dim=-1).squeeze()


class NeuralNet(nn.Module):
    def __init__(self, env: Env, hid_layers = [64, 64],
                action_space=None, net_type='shared', bins=None,
                act_fn='relu', ordinal=False, init_logstd=0.0,
                conv_layers=None, batch_norm=False,
                max_pool = None):
        super(NeuralNet, self).__init__()

        self.create(env, hid_layers, action_space,
                    net_type, bins, act_fn, ordinal,
                    init_logstd, conv_layers,
                    batch_norm, max_pool)
        print(f'feed forward network: {self.feedfwd}')
        if hasattr(self, 'log_std'):
            print(f'log standard deviation parameter: {self.log_std}')

    def forward(self, x):
        logits = self.feedfwd(x)
        return logits

    def create(self, env: Env, hid_layers = [64, 64],
                action_space=None, net_type='shared', bins=None,
                act_fn='relu', ordinal=False, init_logstd = 0.0,
                conv_layers=None, batch_norm=False,
                max_pool = None):
    
        state_shape = env.observation_space.sample().shape
        action_dim = len(env.action_space.sample())

        print("State sample shape:", state_shape)
        print("Action sample shape:", action_dim)
        if len(state_shape) > 1:
            inp_shape = np.prod(state_shape) #its an image
        else:
            inp_shape = state_shape[0] #its an 1d array

        layers = []
        activation_fun = {'relu': nn.ReLU(),
                        'softplus':nn.Softplus(),
                        'tanh':nn.Tanh(),
                        'elu': nn.ELU()}

        if conv_layers is not None:
            inp_h, inp_w, in_chann = inp_shape
            for conv in conv_layers:
                out_chann, filter_size, stride = conv
                layers.append(nn.Conv2d(in_chann, out_chann, filter_size, stride))
                if batch_norm:
                    layers.append(nn.BatchNorm2d(out_chann))
                layers.append(activation_fun[act_fn])

                out_h = (inp_h - filter_size)//stride + 1
                out_w = (inp_w - filter_size)//stride + 1
                inp_h = out_h
                inp_w = out_w
                print(f'h = {inp_h}, w = {inp_w}, c = {in_chann}')
            if max_pool is not None:
                layers.append(nn.MaxPool2d(max_pool[0], max_pool[1]))
                out_h = (inp_h - max_pool[0])//max_pool[1] + 1
                out_w = (inp_w - max_pool[0])//max_pool[1] + 1
                inp_h = out_h
                inp_w = out_w
                print(f'h = {inp_h}, w = {inp_w}, c = {in_chann}')

                in_chann = out_chann
            layers.append(nn.Flatten())
            layers.append(nn.Linear(inp_h*inp_w*in_chann, hid_layers[0]))
            layers.append(activation_fun[act_fn])

        else:
            inp = np.prod(inp_shape)
            layers.append(nn.Linear(inp, hid_layers[0]))
            layers.append(activation_fun[act_fn])

        # try:
        # except:
        #     pass
        
        if len(hid_layers) > 1:
            dim_pairs = zip(hid_layers[:-1], hid_layers[1:])
            for in_dim, out_dim in list(dim_pairs):
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(activation_fun[act_fn])

        if action_space == 'disc':
            out = env.action_space.n
        elif action_space == 'cont':
            out = action_dim
            self.log_std = nn.Parameter(torch.ones(out)*init_logstd, requires_grad  = True)
        elif action_space == 'discretize':
            out = action_dim*bins
        else:
            out = 0
                    
        net_types = {'actor': out, 'critic': 1, 'shared': out+1, 'two_head': 2}
        if net_type in net_types.keys():
            
            layers.append(nn.Linear(hid_layers[-1], net_types[net_type]))
            
            if action_space == 'discretize' and net_type == 'actor':
                self.latent_layer = Ordinal(action_dim, ordinal)



        self.feedfwd =  nn.Sequential(*layers)