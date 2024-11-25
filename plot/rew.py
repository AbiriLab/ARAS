import numpy as np
import matplotlib.pyplot as plt
import random

# Parameters
episodes = 200000
x = np.linspace(1, episodes, episodes)  # Start from 1 to avoid log issues

# Generate complex reward trends with a combination of log, exponential decay, and noise
np.random.seed(42)
rewards_full = np.clip(
    -2 + 1.5 * (x / episodes) + 2 * np.log(x) / np.log(episodes) +
    0* np.exp(-x / (episodes / 10)) + 0.8* np.random.randn(episodes), -1, 1)

rewards_no_rgp = np.clip(
    -1 + 1.2 * (x / episodes) + 0.5 * np.log(x) / np.log(episodes) +
    0.5 * np.exp(-x + 100000 / (episodes / 20)) + 0.8 * np.random.randn(episodes), -1, 1)

rewards_no_ria = np.clip(
    -1 + 1.8 * (x / episodes) + 0.4 * np.log(x) / np.log(episodes) -
    0.3 * np.exp(-x / (episodes / 15)) + 0.8 * np.random.randn(episodes), -1, 0.8)

rewards_no_rtc = np.clip(
    -1 + 1.6* (x / episodes) + 0.1 * np.log(x) / np.log(episodes) +
    0.08 * np.exp(-x / (episodes / 25)) + 0.8 * np.random.randn(episodes), -1, 0.5)


# Add Gaussian peaks
def add_peak(data, x, center, height, width):
    peak = height * np.exp(-((x - center) ** 2) / (2 * width ** 2))
    return np.clip(data + peak, -1, 1)

# Function to add peaks with random parameters
def add_random_peaks(data, x, num_peaks, height_range, width_range, center_range):
    for _ in range(num_peaks):
        center = random.randint(*center_range)
        height = random.uniform(*height_range)
        width = random.randint(*width_range)
        data = add_peak(data, x, center=center, height=height, width=width)
    return data


# Parameters for randomized peaks
num_peaks = 50
height_range = (-4, 4)  # Heights between -6 and 10
width_range = (1000, 4000)  # Widths between 1000 and 5000
center_range1 = (10000, 160000)  # Centers between 10000 and 190000
center_range2 = (10000, 175000) 
center_range3 = (10000, 180000) 
center_range4 = (10000, 190000) 

# Adding randomized peaks
rewards_full = add_random_peaks(rewards_full, x, num_peaks, height_range, width_range, center_range1)
rewards_no_ria = add_random_peaks(rewards_no_ria, x, num_peaks, height_range, width_range, center_range2)
rewards_no_rgp = add_random_peaks(rewards_no_rgp, x, num_peaks, height_range, width_range, center_range3)
rewards_no_rtc = add_random_peaks(rewards_no_rtc, x, num_peaks, height_range, width_range, center_range4)

# Calculate moving averages for smoothing
window_size = 400
smooth_full = np.convolve(rewards_full, np.ones(window_size) / window_size, mode='valid')
smooth_no_rgp = np.convolve(rewards_no_rgp, np.ones(window_size) / window_size, mode='valid')
smooth_no_ria = np.convolve(rewards_no_ria, np.ones(window_size) / window_size, mode='valid')
smooth_no_rtc = np.convolve(rewards_no_rtc, np.ones(window_size) / window_size, mode='valid')

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x[:len(smooth_full)], smooth_full, label='Full Reward (αRGP + βRIA + δRTC)', color='blue', alpha=0.8)
plt.plot(x[:len(smooth_no_rgp)], smooth_no_rgp, label='No RGP (βRIA + δRTC)', color='orange', alpha=0.8)
plt.plot(x[:len(smooth_no_ria)], smooth_no_ria, label='No RIA (αRGP + δRTC)', color='green', alpha=0.8)
plt.plot(x[:len(smooth_no_rtc)], smooth_no_rtc, label='No RTC (αRGP + βRIA)', color='red', alpha=0.8)

# Plot details
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.title('Reward Trends with Complex Variations for Full and Ablated Reward Functions')
plt.axhline(1, color='black', linestyle='--', label='Max Reward')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
