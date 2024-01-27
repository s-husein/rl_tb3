import torch
from torch.distributions import Distribution, Categorical

class MultiCategorical(Distribution):
    def __init__(self, probs, out_dims):
        super().__init__(validate_args=False)
        self.probs = torch.chunk(probs, chunks=out_dims, dim=-1)
        self.dists = [Categorical(probs=prob) for prob in self.probs]

    def log_prob(self, sample):
        return torch.sum(torch.FloatTensor([dist.log_prob(s) for s, dist in zip(sample, self.dists)]), dim=-1)
            
    def sample(self):
        return torch.stack([dist.sample() for dist in self.dists])

    def entropy(self):
        return torch.sum(torch.FloatTensor([dist.entropy() for dist in self.dists]), dim=-1)

