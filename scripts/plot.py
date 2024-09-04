import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/home/user/fyp/src/rl_tb3/plots/rnd_ppo_stg1_pc_plot.txt')



# data.plot(color='blue')
data[1050:2000].rolling(70).mean().plot(color='green')

# plt.plot(data)
# plt.plot(data.rolling(100).mean())

plt.legend().set_visible(False)
plt.xlabel('Episodes', fontsize=20)
plt.ylabel('Average Rewards', fontsize=20)
# plt.yticks([-200, 0, 200, 400, 600, 800, 1000, 1200, 1400], fontsize=13)
plt.xticks(fontsize=13)
# plt.axes().set_xticklabels([0, 200, 400, 600, 800, 1000])
plt.grid(linestyle='--')
plt.show()
