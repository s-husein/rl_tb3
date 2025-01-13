import seaborn as sns
import matplotlib.pyplot as plt

# Data
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]



# Plot
plt.figure(figsize=(10, 8))

plt.plot(x, y, linewidth=3, color='green')
# Add labels and title
plt.xlabel('X-axis Label', fontsize=14)
plt.ylabel('Y-axis Label', fontsize=14)

# Add legend

plt.savefig('ex.png', dpi=500)

# Show plot
plt.show()