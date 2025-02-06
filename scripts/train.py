from nets import make_net
from paths import WORKING_DIR, MISC_DIR
import torch
import cv2 as cv
import numpy as np
from algos import PPO
from drive import GoogleDrive
from gymenv import Gym
from yaml import safe_load


pu = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(f'{WORKING_DIR}/config.yaml') as file:
    params = safe_load(file)

params = {**params['env_params'], **params['network_params'], **params['algo_params']}
g_drive = GoogleDrive(MISC_DIR)


env = params['env'] = Gym(**params)

make_net(params)

agent = PPO(**params)

epoch = agent.check_status()

episodes = params['episodes']+1

for ep in range(epoch, episodes):
    done = False
    # try:
    state = env.reset()[0]
    ep_reward = 0
    steps = 0
    while not done:
        d_s = (np.transpose(state[0], (2, 0, 1))/255.0).astype(np.float32)
        action = agent.act(d_s)
        # try:
        next_state, reward, done, info, _ = env.step(action.cpu().detach().numpy())
        cv.imshow('depth', state[0])
        cv.waitKey(1)
        # except:
            # except_flag = True
            # break
        d_ns = (np.transpose(next_state[0], (2, 0, 1))/255.0).astype(np.float32)
        agent.buffer.add_experience(d_s, action, d_ns, reward, done)
        state = next_state
        ep_reward += reward
        steps += 1
        if steps >= params['max_steps']:
            break
    # except:
    #     ep -= 1
    #     continue

    print(f'ep. {ep}\t{ep_reward = :.3f}\t{steps = }')
    agent.write_plot_data(ep_reward)
    agent.save_check_interval(episodes, epoch=ep, interval=20)
    agent.save_best_model(float(ep_reward))
    agent.train()

if agent.configs['status'] == 'finished':
    g_drive.upload_folder()







