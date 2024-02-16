import torch


x = torch.tensor([1, 2, 3, 4, 5], dtype = torch.float32)

xy = [x for i in range(5)]

xyz = torch.stack(xy)

print((xyz - xyz.mean(dim=-1))/xyz.std(dim=-1))