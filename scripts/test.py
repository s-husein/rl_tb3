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
pred = make_dnn(env, hid_layers=hid_l, net_type='rnd', conv_layers=conv_l, max_pool=max_pool, act_fn='elu').to('cuda')


targ = make_dnn(env, hid_layers=hid_l, net_type='rnd', conv_layers=conv_l, max_pool=max_pool, act_fn='elu').to('cuda')
for param in targ.parameters():
    param.requires_grad = False

# print(pred)

states = torch.stack([torch.tensor(env.reset()[0]).to('cuda') for i in range(5)])

print(states)

rms = RunningMeanStd()

rms.update(states)


print(rms.mean, rms.std, rms.count)

prediction = pred(states)
target = targ(states)

mse_loss = F.mse_loss(prediction, target, reduction='none').mean(dim=-1)

diff = ((target - prediction)**2)

reward = diff.mean(dim=-1).detach()


print(f'reward: {reward}')
print(f'mse_loss: {mse_loss}' )
# rms.update(reward)


# print(reward/rms.std)

