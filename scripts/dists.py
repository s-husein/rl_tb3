import torch
from torch.distributions import Distribution, Categorical

class MultiCategorical(Distribution):
    def __init__(self, probs):
        super().__init__(validate_args=False)
        self._dists = [Categorical(probs=prob) for prob in probs]

    def log_prob(self, sample):
        return torch.stack([dist.log_prob(s) for s, dist in zip(torch.unbind(sample, dim=-1), self._dists)]).sum(dim=-1)
            
    def sample(self):
        return torch.stack([dist.sample() for dist in self._dists], dim=-1)

    def entropy(self):
        return torch.stack([dist.entropy() for dist in self._dists]).sum(dim=-1)

