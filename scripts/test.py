from algos import A2C
import numpy as np
import gym
import torch


env = gym.make('LunarLander-v2', continuous=True)

agent = A2C(env=env, min_batch_size=256, act_space='cont', net_type='actor-critic')


state = env.reset()[0]

for i in range(5):
    action = agent.act(state)
    next_state, reward, done, *others = env.step(action.cpu().detach().numpy())
    agent.buffer.add_experience(state, action, next_state, reward, done)
    state = next_state

agent.train()





