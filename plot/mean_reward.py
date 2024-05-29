import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def extract_mean_rewards_from_file(file_path):
    # Read the content of the log file
    with open(file_path, 'r') as file:
        log_data = file.read()
    
    # Define the regex pattern to match the lines with "Mean Reward"
    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) - Mean Reward at episode (\d+): ([\d\.]+)"
    
    # Find all matches in the log data
    matches = re.findall(pattern, log_data)
    
    # Create a list of dictionaries from the matches
    rewards = [{"Episode": int(match[1]), "Mean Reward": float(match[2])} for match in matches]
    
    return rewards

def plot_mean_rewards(mean_rewards, smoothing_window=10):
    # Convert to DataFrame
    df = pd.DataFrame(mean_rewards)
    
    # Filter for the first 200,000 episodes
    df = df[df['Episode'] <= 200000]
    
    # Set the episode number as the index
    df.set_index('Episode', inplace=True)
    
    # Calculate the smoothed mean reward using a rolling window
    df['Smoothed Mean Reward'] = df['Mean Reward'].rolling(window=smoothing_window, center=True).mean()
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Mean Reward'], color='gray', alpha=0.7, label='Mean Reward')
    plt.plot(df.index, df['Smoothed Mean Reward'], color='black', alpha=0.8, label='Smoothed Mean Reward', linewidth=2)
    
    # Add labels, title, and legend
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward over Episodes (First 200,000)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Specify the path to your log file
log_file_path = '/home/tnlab/Projects/github/sharedcontrol_DQN_Kinova/logs/pickplace_seg_v4_bs64_ss4_rb30000_gamma0.99_decaylf120000_lr0.001.log'

# Extract mean rewards from the log file
mean_rewards = extract_mean_rewards_from_file(log_file_path)

# Plot the mean rewards
plot_mean_rewards(mean_rewards)
