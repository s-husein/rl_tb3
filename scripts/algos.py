from nets import make_dnn
from utils import Utils
from mems import Rollout
from gym import Env
from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from paths import MODELFOLDER, PLOTFOLDER
    

class REINFORCE(Utils):
    def __init__(self, env: Env, name='reinforce', hid_layer = [128, 128], shared_net= True, lr = 0.0003, act_space = 'disc'):
        self.buffer = Rollout()
        self.shared_net = shared_net
        self.model_file = f'{MODELFOLDER}/{name}.pth'
        self.plot_file = f'{PLOTFOLDER}/{name}.txt'
        self.act_space = act_space
        self.check_status_file()
        self.model = make_dnn(env, net_type='actor', action_space=act_space)
        self.optim = Adam(self.model.parameters(), lr = lr, eps = 1e-8)
        self.model.train()

    def act(self, state):
        state = torch.from_numpy(state)
        probs = F.softmax(self.model(state))
        if self.act_space == 'disc':
            dist = Categorical(probs=probs)
        action = dist.sample()
        return action
    

    def discounted_rewards(self, rewards, dones):
        g = 0.0
        not_dones = 1 - np.array(dones)
        T = len(rewards)
        ret = np.empty(T, dtype=np.float32)
        for t in reversed(range(T)):
            g = rewards[t] + g*0.999*not_dones[t]
            ret[t] = g
        ret = (ret - ret.mean())/ret.std()
        return ret
    

    def train(self):
        self.write_plot_data(sum(self.buffer.traj['rewards']))
        batch =  self.buffer.sample()
        states = torch.stack([self.buffer.traj['states'][i] for i in batch])
        actions = torch.stack([self.buffer.traj['actions'][i] for i in batch])
        probs = F.softmax(self.model(states))
        adv = torch.tensor(self.discounted_rewards(rewards=[self.buffer.traj['rewards'][i] for i in batch], dones=[self.buffer.traj['dones'][i] for i in batch]))
        if self.act_space == 'disc':
            dist = Categorical(probs=probs)

        log_probs = dist.log_prob(actions)
        loss = (-log_probs * adv).sum()
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optim.step()
        self.buffer.reset()

