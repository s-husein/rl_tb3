import torch
import numpy as np
from torch.distributions import Categorical

from dists import MultiCategorical
x = torch.tensor([-2.2912,  0.7968, -1.6372, -0.6192, -0.0620,  0.5913,  0.6419,  0.2147,
                  0.1110, -0.6814, -1.9464,  1.8108, -0.6366, -0.4581])

x = torch.softmax(torch.sigmoid(x), dim=-1)



dist = MultiCategorical(probs=x, out_dims = 2)

action = dist.sample()

print(action)

print(dist.log_prob(action))
# print(logits)

# actions = np.linspace(-1, 1, len(logits[0]))
# print(actions)

# dists = [Categorical(probs = torch.softmax(logit, dim=0)) for logit in logits]

# action = [dist.sample().item() for dist in dists]


# print(action)