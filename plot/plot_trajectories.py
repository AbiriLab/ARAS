import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_trajectory_data(file_path):
    """
    Load trajectory data from a JSON file.
    Args:
        file_path: Path to the JSON file.
    Returns:
        List of trajectory information.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def plot_trajectories(data, save_path=None):
    """
    Plot gripper trajectories along with object and bin positions.
    Args:
        data: List of trajectory information loaded from JSON.
        save_path: Path to save the plot image (optional).
    """
    plt.figure(figsize=(8, 8))

    # Loop through episodes in the data
    for episode_data in data:
        trajectory = np.array(episode_data["trajectory"])  # Convert to NumPy array
        goal_position = episode_data["goal_position"]
        bin_position = episode_data["bin_position"]

        # Plot the trajectory
        plt.plot(
            trajectory[:, 0], trajectory[:, 1], 
            label=f"Episode {episode_data['episode']}", 
            alpha=0.7
        )

        # Mark the object and bin positions
        plt.scatter(goal_position[0], goal_position[1], color='red', marker='x', s=100, label="Object")
        plt.scatter(bin_position[0], bin_position[1], color='blue', marker='o', s=100, label="Bin")

    # Avoid duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Add labels, legend, and grid
    plt.title("Gripper Trajectories")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid()

    # Save or show the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
        plt.savefig(save_path)
    plt.show()

# Specify the file path to the saved trajectory data
file_path = "./trajectory_train_data/trajectory_data_12000.json"  # Replace with your file path
trajectory_data = load_trajectory_data(file_path)

# Plot the trajectories
plot_trajectories(trajectory_data, save_path="./trajectory_train_data/trajectory_plot.png")
