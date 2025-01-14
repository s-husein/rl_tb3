from nets import make_net
from paths import WORKING_DIR, MISC_DIR
import torch
import cv2 as cv
import numpy as np
from algos import PPO
from drive import GoogleDrive
# from gymenv import Gym
import gymnasium as gym
from yaml import safe_load


pu = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(f'{WORKING_DIR}/config.yaml') as file:
    params = safe_load(file)

env = params['env'] = gym.make("BipedalWalker-v3", render_mode="rgb_array")

make_net(params)

g_drive = GoogleDrive(MISC_DIR)

agent = PPO(**params['algo_params'], **params['network_params'])

epoch = agent.check_status()
episodes = params['algo_params']['episodes']+1

for ep in range(epoch, episodes):
    done = False
    total_rewards = 0
    steps = 0
    state = env.reset()[0]
    while not done:
        action = agent.act(state)
        
        next_state, reward, done, info, _ = env.step(action.cpu().numpy())

        agent.buffer.add_experience(state, action, next_state, reward, done)
        
        state = next_state

        steps += 1
        total_rewards += reward
        if steps >= params['algo_params']['max_steps']:
            done = True

    print(f'ep. {ep}\t{total_rewards = :.3f}\t{steps = }')
    agent.write_plot_data(total_rewards)
    agent.save_check_interval(epoch=ep, interval=10)
    agent.save_best_model(float(total_rewards))
    agent.train()
g_drive.upload_folder()





