import torch

device ='cuda'

update_proportion = 0.25

forward_loss = torch.rand(10).to(device)

print(forward_loss)

mask = torch.rand(len(forward_loss)).to(device)

print(mask)
mask = (mask < update_proportion).type(torch.FloatTensor).to(device)
print(mask)

print(forward_loss * mask)
forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(device))

print(forward_loss)