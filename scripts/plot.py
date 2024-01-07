import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/home/user/fyp/src/rl_tb3/plots/reinforce.txt')


print(data.max())
data.plot()


plt.show()
