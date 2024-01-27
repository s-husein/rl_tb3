import torch
from torch.distributions import Distribution, Categorical

class MultiCategorical(Distribution):
    def __init__(self, probs, out_dims):
        super().__init__(validate_args=False)
        self.probs = torch.chunk(probs, chunks=out_dims, dim=-1)
        self.dists = [Categorical(probs=prob) for prob in self.probs]

    def log_prob(self, sample):
        log_probs = []
        for s, dist in zip(sample, self.dists):
            print(s)
            log_probs.append(dist.log_prob(s))
            

        return torch.sum(torch.FloatTensor(log_probs), dim=-1)

        # return torch.sum([dist.log_prob(s) for s, dist in zip(sample, self.dists)], dim=-1)

    def sample(self):
        return torch.tensor([dist.sample() for dist in self.dists])

    def entropy(self):
        return torch.sum([dist.entropy() for dist in self.dists], dim=-1)

