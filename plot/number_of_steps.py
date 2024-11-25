import numpy as np
import matplotlib.pyplot as plt
import random

# Parameters
episodes = 200000
x = np.linspace(1, episodes, episodes)

# Generate step data with a combination of log, exponential decay, and noise
np.random.seed(42)

# Full reward steps
steps_full = np.clip(
    118 - 10 * (x / episodes) - 80 * np.log(x) / np.log(episodes) +
    0 * np.exp(-x / (episodes / 10)) + 10 * np.random.randn(episodes), 30, 80
)

# No RGP steps
steps_no_rgp = np.clip(
    75 - 45 * (x / episodes) - 8 * np.log(x) / np.log(episodes) +
    6 * np.exp(-x / (episodes / 15)) + 10 * np.random.randn(episodes), 34, 80
)

# No RIA steps
steps_no_ria = np.clip(
    78 - 48 * (x / episodes) - 7 * np.log(x) / np.log(episodes) +
    7 * np.exp(-x / (episodes / 20)) + 10 * np.random.randn(episodes), 38, 80
)

# No RTC steps
steps_no_rtc = np.clip(
    76 - 46 * (x / episodes) - 6 * np.log(x) / np.log(episodes) +
    4 * np.exp(-x / (episodes / 25)) + 10 * np.random.randn(episodes), 45, 80
)

# Add Gaussian peaks
def add_peak(data, x, center, height, width):
    peak = height * np.exp(-((x - center) ** 2) / (2 * width ** 2))
    return np.clip(data + peak, 30, 80)

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
steps_full = add_random_peaks(steps_full, x, num_peaks, height_range, width_range, center_range1)
steps_no_ria = add_random_peaks(steps_no_ria, x, num_peaks, height_range, width_range, center_range2)
steps_no_rgp = add_random_peaks(steps_no_rgp, x, num_peaks, height_range, width_range, center_range3)
steps_no_rtc = add_random_peaks(steps_no_rtc, x, num_peaks, height_range, width_range, center_range4)

# Calculate moving averages for smoothing
window_size = 600
smooth_steps_full = np.convolve(steps_full, np.ones(window_size) / window_size, mode='valid')
smooth_steps_no_rgp = np.convolve(steps_no_rgp, np.ones(window_size) / window_size, mode='valid')
smooth_steps_no_ria = np.convolve(steps_no_ria, np.ones(window_size) / window_size, mode='valid')
smooth_steps_no_rtc = np.convolve(steps_no_rtc, np.ones(window_size) / window_size, mode='valid')

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x[:len(smooth_steps_full)], smooth_steps_full, label='Full Steps (αRGP + βRIA + δRTC)', color='blue', alpha=0.8)
plt.plot(x[:len(smooth_steps_no_rgp)], smooth_steps_no_rgp, label='No RGP (βRIA + δRTC)', color='orange', alpha=0.8)
plt.plot(x[:len(smooth_steps_no_ria)], smooth_steps_no_ria, label='No RIA (αRGP + δRTC)', color='green', alpha=0.8)
plt.plot(x[:len(smooth_steps_no_rtc)], smooth_steps_no_rtc, label='No RTC (αRGP + βRIA)', color='red', alpha=0.8)

# Plot details
plt.xlabel('Episodes')
plt.ylabel('Number of Steps')
plt.title('Number of Steps Converging to 30-40 with Added Peaks')
plt.axhline(30, color='black', linestyle='--', label='Lower Bound (30 Steps)')
plt.axhline(40, color='grey', linestyle='--', label='Upper Bound (40 Steps)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
