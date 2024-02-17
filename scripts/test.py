from nets import make_dnn
from gymenv import Gym
import torch
import torch.nn.functional as F
import numpy as np

hid_l = [64, 10]
conv_l = [[16, 3, 1],
          [32, 3, 1],
          [64, 3, 1]]
max_pool = [2, 2]

env = Gym(conv_layers=conv_l, obs_scale_factor=0.1, positions=[(4, -5)])
pred = make_dnn(env, hid_layers=hid_l, net_type='rnd', conv_layers=conv_l, max_pool=max_pool, act_fn='elu').to('cuda')


targ = make_dnn(env, hid_layers=hid_l, net_type='rnd', conv_layers=conv_l, max_pool=max_pool, act_fn='elu').to('cuda')
for param in targ.parameters():
    param.requires_grad = False

print(pred)

states = torch.stack([torch.tensor(env.reset()[0]).to('cuda') for i in range(5)])


prediction = pred(states)
target = targ(states)

reward = F.mse_loss(prediction, target, reduction='none').mean(dim=-1)

count = len(reward)

_mean = np.zeros(count, dtype = np.float32)
_var = np.ones(count, dtype = np.float32)
_count = 1e-4

batch_mean = torch.mean(reward)
batch_var = torch.var(reward)


new_count = count + _count
mean_delta = batch_mean - _mean
new_mean = _mean + mean_delta * count / new_count
# # this method for calculating new variable might be numerically unstable
# m_a = self._var * self._count
# m_b = batch_var * batch_count
# m2 = m_a + m_b + np.square(mean_delta) * self._count * batch_count / new_count
# new_var = m2 / new_count
# self._mean = new_mean
# self._var = new_var
# self._count = new_count

print(reward)
print(reward.shape)
print(_mean)
print(_var)

print(batch_mean)
print(batch_var)

