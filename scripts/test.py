from nets import make_dnn
from gymenv import Gym
import torch
import torch.nn.functional as F


hid_l = [64, 10]
conv_l = [[16, 3, 1],
          [32, 3, 1],
          [64, 3, 1]]
max_pool = [2, 2]

env = Gym(conv_layers=conv_l, obs_scale_factor=0.1)
pred = make_dnn(env, hid_layers=hid_l, net_type='rnd', conv_layers=conv_l, max_pool=max_pool, act_fn='elu').to('cuda')


targ = make_dnn(env, hid_layers=hid_l, net_type='rnd', conv_layers=conv_l, max_pool=max_pool, act_fn='elu').to('cuda')
for param in targ.parameters():
    param.requires_grad = False

print(pred)

states = torch.stack([torch.tensor(env.reset()[0]).to('cuda') for i in range(5)])


prediction = pred(states)
target = targ(states)

print(F.mse_loss(prediction, target, reduction='none').sum(dim=-1))

