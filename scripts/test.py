from nets import make_net
from paths import WORKING_DIR, MISC_DIR
import torch
import cv2 as cv
import numpy as np
from algos import PPO
from drive import GoogleDrive
from gymenv import Gym
import gymnasium as gym
from yaml import safe_load


pu = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(f'{WORKING_DIR}/config.yaml') as file:
    params = safe_load(file)

params = {**params['env_params'], **params['network_params'], **params['algo_params']}


env = params['env'] = Gym(**params)

make_net(params)

g_drive = GoogleDrive(MISC_DIR)

agent = PPO(**params)

epoch = agent.check_status()

episodes = params['episodes']+1


state = env.reset()[0][0]

state = state.flatten()

state = torch.from_numpy(state/255).float().to(pu)

print(agent.old_policy(state))





# for ep in range(epoch, episodes):
#     except_flag = False
#     done = False
#     # try:
#     state = env.reset()[0][0]
#     # except:
#     #     ep -= 1
#     #     continue
#     ep_reward = 0
#     steps = 0
#     while not done:
#         action = agent.act(state/255)
#         try:
#             next_state, reward, done, info, _ = env.step(action.cpu().detach().numpy())
#             env.render()
#         except:
#             except_flag = True
#             break
#         agent.buffer.add_experience(state, action, next_state, reward, done)
#         state = next_state
#         ep_reward += reward
#         steps += 1
#         if steps >= params['max_steps']:
#             done = True
#     if except_flag:
#         ep -= 1
#         continue

#     print(f'ep. {ep}\t{ep_reward = :.3f}\t{steps = }')
#     agent.write_plot_data(ep_reward)
#     agent.save_check_interval(epoch=ep, interval=10)
#     agent.save_best_model(float(ep_reward))
#     agent.train()
# g_drive.upload_folder()





