import torch
import numpy as np
from torch.distributions import Categorical

from dists import MultiCategorical
import gym
from nets import make_dnn

env = gym.make('LunarLanderContinuous-v2')

actor = make_dnn(env, action_space='discretize', net_type='actor', bins=5, ordinal=True)

states = []
actions = []

for i in range(1):
    state = torch.tensor(env.reset()[0])
    probs = actor(state)
    print(probs)
    states.append(state)
dist = MultiCategorical(probs)
    # action = dist.sample()
    # print(action)
    # print(torch.unbind(action, dim=-1))
    # print(dist.log_prob(action))

states_ = torch.stack(states)
print(states_)


probs_ = actor(states_)
dist = MultiCategorical(probs_)
action = dist.sample()

print(action)
print(action.unbind(dim=-1))
print(dist.log_prob(action))
print(dist.entropy())

print(probs_)


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