import numpy as np
import torch

class Rollout:
    def __init__(self):
        self.data_keys = ['states', 'actions', 'next_states', 'rewards', 'dones']
        self.reset()
        self.size = 0

    def sample(self):
        indices = np.random.choice(range(self.size), self.size, replace=False)
        return indices
        
    def reset(self):
        self.traj = {k: [] for k in self.data_keys}
        self.size = 0

    def add_experience(self, state, action, next_state, reward, done):
        experience = (torch.tensor(state), action, torch.tensor(next_state), reward, done)
        for idx, key in enumerate(self.data_keys):
            self.traj[key].append(experience[idx])
        self.size += 1

