import torch
import numpy as np

# logits = torch.tensor([ 0.0550, -0.0512,  0.1334, -0.0857, -0.1452,  0.0177])

logits = torch.tensor([[0.5458, 0.5005],
                        [0.5388, 0.4999],
                        [0.5455, 0.5011],
                        [0.5450, 0.5040],
                        [0.5415, 0.4975],
                        [0.5409, 0.5012],
                        [0.5451, 0.5008],
                        [0.5450, 0.5031],
                        [0.5427, 0.4990],
                        [0.5472, 0.5004]])


# logits = torch.sigmoid(torch.tensor([[ 0.0550, -0.0512, 0.0435, -0.0463],
#                        [ 0.0435, -0.0463, 0.0435, -0.0463]]))

# logits = torch.sigmoid(logits)

dims = logits.dim()

iden = torch.tril(torch.ones_like(torch.eye(logits.size()[-1])))[np.newaxis, :, :].repeat(logits.size()[0], 1, 1)
print(iden)
inv_iden = 1 - iden
logits = logits[:, np.newaxis]
print(inv_iden)

print(iden*(logits) + inv_iden*(1-logits))

res = torch.softmax(torch.sum(iden*torch.log(logits) + inv_iden*torch.log(1-logits), dim=2).squeeze(), dim=dims-1)

print(res)
