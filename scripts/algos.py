from nets import MLP
from mems import Rollout
from gym import Env
from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class PPO(MLP):
    def __init__(self, env, hid_layers):
        super(MLP, self).__init__()
        


    def train(self):
        pass

class REINFORCE:
    def __init__(self, env: Env, lr = 0.0003, batch_size = 256, act_space = 'disc'):
        self.buffer = Rollout(batch_size=batch_size)
        self.act_space = act_space
        self.model = MLP(env = env, act_space=act_space)
        self.optim = Adam(self.model.parameters(), lr = lr, eps = 1e-5)
        self.model.train()

    def act(self, state):
        state = torch.from_numpy(state)
        probs = F.softmax(self.model(state))
        if self.act_space == 'disc':
            dist = Categorical(probs=probs)
        action = dist.sample()
        return action
    

    def discounted_rewards(self, rewards):
        g = 0.0
        T = len(rewards)
        ret = np.empty(T, dtype=np.float32)
        for t in reversed(range(T)):
            g = rewards[t] + g*0.999
            ret[t] = g
        ret = (ret - ret.mean())/ret.std()
        return ret
    

    def train(self):
        if self.buffer.size >= self.buffer.batch_size:
            batch =  self.buffer.sample()
            states = torch.stack([self.buffer.traj['states'][i] for i in batch])
            actions = torch.stack([self.buffer.traj['actions'][i] for i in batch])
            probs = F.softmax(self.model(states))
            adv = torch.tensor(self.discounted_rewards(rewards=[self.buffer.traj['rewards'][i] for i in batch]))
            if self.act_space == 'disc':
                dist = Categorical(probs=probs)

            log_probs = dist.log_prob(actions)
            loss = (-log_probs * adv).sum()
            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optim.step()
            self.buffer.reset()
        else:
            pass

