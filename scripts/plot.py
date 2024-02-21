import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/home/user/fyp/src/rl_tb3/plots/pppo.txt')



data.rolling(30).mean().plot()


plt.show()
