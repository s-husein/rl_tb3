import torch
import numpy as np
from torch.distributions import Categorical

from dists import MultiCategorical
import gym
from nets import make_dnn

env = gym.make('LunarLanderContinuous-v2')

actor = make_dnn(env, action_space='discretize', net_type='actor', bins=5, ordinal=False)


state = torch.tensor(env.reset()[0])

print(actor(state))