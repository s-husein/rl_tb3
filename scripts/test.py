import torch
import numpy as np


logits = torch.rand(4)

iden = torch.tril(torch.ones(2*logits.size()))
inv_iden = 1-iden

res = torch.sum(iden*torch.log(logits)+inv_iden*torch.log(1-logits), dim=1)
print(logits)

print(res)

print(torch.softmax(res, dim=0))
