import numpy as np
import torch
import random


class Rollout:
    def __init__(self):
        self.data_keys = ['states', 'actions', 'next_states', 'rewards', 'dones']
        self.reset()
        self.size = 0

    def sample(self):
        indices = np.arange(self.size)
        # indices = np.random.choice(range(self.size), self.size, replace=False)
        return indices
        
    def reset(self):
        self.traj = {k: [] for k in self.data_keys}
        self.size = 0

    def get_mini_batches(self, mb_size):

        indices = np.arange(self.size)
        # np.random.shuffle(indices)

        mini_bts = round(self.size/mb_size)
        mini_batches = []
        ind = 0
        for i in range(mini_bts-1):
            mini_batches.append(indices[ind: ind+mb_size])
            ind += mb_size
        mini_batches.append(indices[ind:])
        random.shuffle(mini_batches)
        return mini_batches

    def add_experience(self, state, action, next_state, reward, done):
        experience = (torch.tensor(state), action, torch.tensor(next_state), reward, done)
        for idx, key in enumerate(self.data_keys):
            if experience[idx] is not None:
                self.traj[key].append(experience[idx])
        self.size += 1

    def augment(self):
        len = self.size
        inv = torch.tensor([1, -1]).to('cuda')
        for i in range(len):
            self.traj['states'][i] = self.traj['states'][i].flip(2)
            self.traj['actions'][i] = self.traj['actions'][i]*inv
            self.traj['next_states'][i] = self.traj['next_states'][i].flip(2)
        print('augmented')
