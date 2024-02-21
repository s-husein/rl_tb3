import numpy as np
import torch

class Rollout:
    def __init__(self):
        self.data_keys = ['states', 'actions', 'next_states', 'rewards', 'dones', 'in_rewards']
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

        return mini_batches

    def add_experience(self, state, action, next_state, reward, done, in_rewards):
        experience = (torch.tensor(state), action, torch.tensor(next_state), reward, done, in_rewards)
        for idx, key in enumerate(self.data_keys):
            self.traj[key].append(experience[idx])
        self.size += 1

