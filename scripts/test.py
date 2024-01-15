import torch 
import numpy as np


def create_tensor(x):
    with torch.no_grad():
        return torch.tensor(x, requires_grad=True)


x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y = create_tensor(x)

print(y)