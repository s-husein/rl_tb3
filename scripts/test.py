import torch
import numpy as np

# logits = torch.tensor([ 0.0550, -0.0512,  0.1334, -0.0857, -0.1452,  0.0177])

logits = torch.sigmoid(torch.tensor([[ 0.0550, -0.0512, 0.0435, -0.0463],
                       [ 0.0435, -0.0463, 0.0435, -0.0463]]))

dims = logits.dim()

print(dims)
# print(logits.size())

iden = torch.tril(torch.ones_like(torch.eye(logits.size()[-1])))[np.newaxis, :, :].repeat(dims, 1, 1)
inv_iden = 1 - iden

logits = logits[:, np.newaxis]


res = iden*logits

print(res)

print(torch.sum(res, dim=2).squeeze())


# print(torch.sum(iden*logits, dim=1))

# iden = torch.stack(logits.size()[0]* (torch.tril(torch.ones_like(torch.eye(logits.size(logits.dim()-1)))),))


# print(iden*logits)

# inv_iden = 1-iden

# print(inv_iden)
# res = torch.sum(iden*torch.log(logits)+inv_iden*torch.log(1-logits), dim=1)
# print(logits)

# print(res)

# print(torch.softmax(res, dim=0))
