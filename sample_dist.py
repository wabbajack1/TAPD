import numpy as np
import matplotlib.pyplot as plt

# Define smooth step functions using sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def task1(x):
    return sigmoid(10 * (x - 1)) - sigmoid(10 * (x - 3))

def task2(x):
    return sigmoid(10 * (x - 3)) - sigmoid(10 * (x - 5))

def task3(x):
    return sigmoid(10 * (x - 5)) - sigmoid(10 * (x - 7))

# Generate data for the plot
x = np.linspace(0, 8, 1000)
y1 = task1(x)
y2 = task2(x)
y3 = task3(x)

# Normalize the tasks to make their sum equal to 1 at each point
sum_tasks = y1 + y2 + y3
y1_normalized = y1 / sum_tasks
y2_normalized = y2 / sum_tasks
y3_normalized = y3 / sum_tasks

# Sample from the distribution
environments = ['PongNoFrameskip-v4', 'HeroNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4']
num_samples = 1

for i in range(1000):
    samples = np.random.choice(environments, num_samples, p=[y1_normalized[i], y2_normalized[i], y3_normalized[i]])
    # Print the sampled environments
    print("Sampled environments:", samples)


# Colorblind-friendly colors
colors = ['#0072B2', '#D55E00', '#009E73']

# Plot the smooth step functions with normalized probabilities
plt.plot(x, y1_normalized, label="Task1:" + environments[0], color=colors[0])
plt.plot(x, y2_normalized, label="Task2:" + environments[1], color=colors[1])
plt.plot(x, y3_normalized, label="Task3:" + environments[2], color=colors[2])

plt.xlabel("x")
plt.ylabel("Probability of Task Being Taken")
plt.legend()
plt.title("Smooth Transition Between Tasks")
plt.show()
