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

for i in range(5):
    state = torch.tensor(env.reset()[0])
    probs = actor(state)
    dist = MultiCategorical(probs)
    action = dist.sample()
    print(action)
    states.append(state)
    actions.append(action)

states_ = torch.stack(states)
actions_= torch.stack(actions)
print(actions_)

print(actions_[3])

print(actions_.unbind(dim=-1))

probs_ = actor(states_)

dist_ = MultiCategorical(probs_)

log_prob = dist_.log_prob(actions_)

print(log_prob)