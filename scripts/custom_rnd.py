import torch.nn as nn
from gymenv import Gym
import torch
import numpy as np
from nets import make_dnn


class NeuralNet(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 3)
        self.conv2 = nn.Conv2d(16, 32, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.max_pool = nn.MaxPool2d(2, 2)
        self.l1 = nn.Linear(1280+1, 256)
        self.l2 = nn.Linear(256, 256)
        if type == 'value':
            self.out = nn.Linear(256, 2)
        elif type == 'policy':
            self.out = nn.Linear(256, 4)

    def forward(self, x, y):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(self.max_pool(x))
        x = torch.cat((x, y), dim=1)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.out(x)
        return x


policy = NeuralNet('policy')
values = NeuralNet('value')

pred_net = None
targ_net = None

env = Gym(action_space='cont', obs_scale_factor=0.1, conv_layers=True)


_, depth = env.get_observation()

depth = np.expand_dims(depth, (0, 1))

print(depth.shape)

y = torch.tensor([0.25]).unsqueeze(0)
depth_ = torch.tensor(depth/255.0, dtype=torch.float32)

output = policy(depth_, y)

print(output)
