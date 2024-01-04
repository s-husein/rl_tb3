import torch as th
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from gym import Env
from gymenv import Gym
import gym 


th.device = 'cuda' if th.cuda.is_available() else 'cpu'
print(f'using {th.device}')

class CustomActiv(th.nn.Module):
    def forward(self, x):
        return th.cat((F.tanh(x[:(len(x)//2)]), th.clip(F.sigmoid(x[-(len(x)//2):]), 0.08, 0.3)))

class FCL(th.nn.Module):
    def __init__(self):
        super(FCL, self).__init__()
        self.activation = {'relu': th.nn.ReLU(), 'softmax':th.nn.Softmax(dim=0), 'sigmoid':th.nn.Sigmoid(), 'tanh': th.nn.Tanh()}
        self.env:Env = None
        self.layers = []
        self.hid_lay = None
        self.act_func = None

    def layer_decorator(func):
        def wrapper(self, *args, **kwards):
            calling_class = self._get_name()
            if calling_class == 'MLP':
                inp = np.prod(self.env.observation_space.shape)
            self.layers.append(th.nn.Linear(inp, self.hid_lay[0]))
            self.layers.append(self.act_func)
            func(self, *args, **kwards)
            out_dim = self.env.action_space.sample()
            if isinstance(out_dim, np.ndarray):
                self.layers.append(th.nn.Linear(self.hid_lay[-1], 2*len(out_dim)))
            else:
                self.layers.append(th.nn.Linear(self.hid_lay[-1], self.env.action_space.n))
        return wrapper
    
    @layer_decorator
    def make_fc_layers(self):
        dim_pairs = list(zip(self.hid_lay[:-1], self.hid_lay[1:]))
        for in_dim, out_dim in dim_pairs:
            self.layers.append(th.nn.Linear(in_dim, out_dim))
            self.layers.append(self.act_func)



class MLP(FCL):
    def __init__(self, env:Env, hid_lay: list=[64,64], act_func = 'relu', act_space='disc'):
        super(MLP, self).__init__()
        self.act_space = act_space
        self.env = env
        self.act_func = self.activation[act_func]
        self.hid_lay = hid_lay
        self.make_fc_layers()
        self.model = th.nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.model(x)
        return x


# env = Gym(disc_action=True)
# env = gym.make('LunarLander-v2', continuous=False)



# agent = MLP(env=env, act_space='cont')

# state = env.reset()[0]

# print(agent.parameters())

# print(agent(th.tensor(state)))




