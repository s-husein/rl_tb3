import torch
import numpy as np



def adv_nstep(values, next_values, rewards, n):
        rets = np.zeros_like(rewards)
        future_ret = next_values
        for t in reversed(range(n)):
            rets[t] = future_ret = rewards[t] + 0.99*future_ret

        advs = rets - values
        target_values = torch.tensor(rets)
        return torch.tensor(advs), target_values






