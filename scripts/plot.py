import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/home/user/fyp/src/rl_tb3/plots/ppo_bipdel_plot.txt')


print(data.max())
data.plot()


plt.show()
