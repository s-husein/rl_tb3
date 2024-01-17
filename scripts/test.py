import random
import numpy as np
batch = [i for i in range(10)]

# np.random.shuffle(batch)
mini_bs = 7

mini_bts = round(len(batch) /mini_bs)
mini_batch = []

ind = 0
for i in range(mini_bts-1):
    mini_batch.append(batch[ind: ind+mini_bs])
    ind += mini_bs
mini_batch.append(batch[ind:])



# for i in range(0, len(batch), mini_bs):
#     mini_batch.append(batch[i:i+mini_bs])

# if len(mini_batch[-1]) < mini_bs/2:
#     mini_batch[-2] += mini_batch[-1]

#     del mini_batch[-1]
print(batch)
print(mini_batch)