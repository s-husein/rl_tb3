import torch
import numpy as np
from torch.distributions import Categorical

from dists import MultiCategorical
import gym
from nets import make_dnn

env = gym.make('LunarLanderContinuous-v2')

actor = make_dnn(env, action_space='discretize', net_type='actor', bins=3, ordinal=True)

states = []
actions = []

for i in range(2):
    state = torch.tensor(env.reset()[0])
    probs = actor(state)
    print(probs)
    # dist = MultiCategorical(probs, out_dims=2)
    # action = dist.sample()
    # print(action)
    # print(torch.unbind(action, dim=-1))
    # print(dist.log_prob(action))
    states.append(state)

states_ = torch.stack(states)
print(states_)

print(actor(states_))


# print(dist.sample())

# x = torch.softmax(torch.sigmoid(x), dim=-1)



# dist = MultiCategorical(probs=x, out_dims = 2)

# action = dist.sample()

# print(action)

# print(dist.log_prob(action))
# print(dist.entropy())
# print(logits)

# actions = np.linspace(-1, 18, len(logits[0]))
# print(actions)

# dists = [Categorical(probs = torch.softmax(logit, dim=0)) for logit in logits]

# action = [dist.sample().item() for dist in dists]


# print(action)