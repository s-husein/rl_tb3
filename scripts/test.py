from nets import make_net
import torch
import cv2 as cv
import numpy as np
from gymenv import Gym
import gymnasium as gym
from yaml import safe_load


pu = 'cuda' if torch.cuda.is_available() else 'cpu'


with open('../config.yaml') as file:
    params = safe_load(file)

print(params)


env = params['env'] = gym.make("BipedalWalker-v3", render_mode="human")

print(params)


make_net(params)

print(params)

# k_epochs = 10
# batch_size = 128
# pi_hid_layers = [64, 64]
# min_batch_size = 2048
# pi_conv_layers = [[16, 5, 1],
#                   [32, 3, 1],
#                   [64, 3, 1]]
# episodes = 10000
# lam = 0.95
# gamma = 0.99
# actor_lr = 3e-5
# critic_lr = 7e-5
# act_space = 'cont'
# name = 'rnd_ppo_stg1_pc'
# std_min_clip =  0.1
# eps_clip= 0.4
# beta = 0.1
# max_pool = [2, 2]
# max_steps = 10000
# obs_scale_factor = 0.1


# model = NeuralNet(env=env,
#                 hid_layers=pi_hid_layers,
#                 action_space=act_space,
#                 net_type='actor',
#                 init_logstd=1.0)


# state = env.observation_space.sample()

# print(model(torch.tensor(state)))



# print(pi)
# img = cv.imread('/home/user/fyp/src/rl_tb3/depht_image.png')
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# img = cv.resize(img, (0, 0), fx = 0.1, fy = 0.1)
# img = img[:int(36*0.75), :]
# img = cv.resize(img, (0, 0), fx = 10, fy = 10)


# cv.imwrite('rescaled_depth_cropped.png', img)



# img = np.expand_dims(img, (0,1))

# img_t = torch.tensor(img/255, dtype=torch.float32).to(pu)

# start = time.time()
# outs = pi(img_t)
# end = time.time()


# print(end - start)

