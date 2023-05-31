import numpy as np
import matplotlib.pyplot as plt

def normal_pdf(x, mean, std_dev):
    """Calculate the normal probability density function for given x, mean, and std_dev."""
    return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))

# Define the mean and standard deviation for the delta wave distribution
mean = 2.25
std_dev = 0.7

# Create a range of frequency values from 0 to 5 Hz with a step of 0.01 Hz
x = np.arange(0, 5, 0.01)

# Calculate the probability density function (PDF) for the normal distribution
pdf = normal_pdf(x, mean, std_dev)

# Plot the delta wave distribution
plt.plot(x, pdf)
plt.title("Delta Wave Distribution (0.5 - 4 Hz)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Probability Density")
plt.show()

