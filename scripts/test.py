from nets import make_net
from paths import WORKING_DIR, MISC_DIR
import torch
import torch.nn as nn
from torchvision import models
import cv2 as cv
import numpy as np
from algos import PPO
from drive import GoogleDrive
from gymenv import Gym
from yaml import safe_load
import datetime as dt






pu = 'cuda' if torch.cuda.is_available() else 'cpu'




with open(f'{WORKING_DIR}/config.yaml') as file:
    params = safe_load(file)

params = {**params['env_params'], **params['network_params'], **params['algo_params']}
env = params['env'] = Gym(**params)


state = env.reset()[0][0]


# print(state.shape)





# # g_drive = GoogleDrive(MISC_DIR)


# start = dt.datetime.now()
# mask = np.full(state.shape, 0b11110000)

# state = np.bitwise_and(state, mask).astype(np.uint8)

# state = np.stack([state, state, state])

# state = torch.tensor(state/255.0).float().permute((3, 0, 1, 2)).to(pu)

# state = torch.tensor(state/255.0)


# result = model(state)

# end = dt.datetime.now()

# print(end - start)

# print(result)

# cv.imshow('x', state)
# cv.waitKey()

# cv.destroyAllWindows()









