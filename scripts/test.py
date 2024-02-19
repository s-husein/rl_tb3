from nets import make_dnn
from gymenv import Gym
import torch
import torch.nn.functional as F
import numpy as np
from utils import RunningMeanStd

hid_l = [64, 10]
conv_l = [[16, 3, 1],
          [32, 3, 1],
          [64, 3, 1]]
max_pool = [2, 2]

rms = RunningMeanStd()

env = Gym(conv_layers=conv_l, obs_scale_factor=0.1, positions=[(4, -5), (4, -6)])
critic = make_dnn(env, hid_layers=hid_l, net_type='two_head', conv_layers=conv_l, max_pool=max_pool, act_fn='elu').to('cuda')

print(critic)
states = torch.stack([torch.tensor(env.reset()[0]).to('cuda') for i in range(5)])



print(states.shape)

values = critic(states).transpose(0, 1)

print(values)
print(values[0])
print(values[1])

# values = torch.unbind(values, dim=-1)

# print(values[0])
# print(values[1])
# print(reward/rms.std)

