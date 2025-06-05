"""
Computes mean and std success rates from a batch of seeded evaluation runs.

Usage:
python compute_success_rate.py [your log directory]/[env name]/[mobilization method name]/[manip policy name]

Usage Example:
python compute_success_rate.py logs/CloseDrawer/lelan/bc_xfmr

@yjy0625
"""
import os
import numpy as np
import re
import argparse

# Function to extract the seed number from folder names
def extract_seed(folder_name):
    match = re.search(r"seed(\d+)", folder_name)
    return int(match.group(1)) if match else None

# Function to compute average success rate for each seed
def compute_success_rate(directory):
    seed_success = {}
    seed_counts = {}

    # Iterate through all subdirectories
    for folder in sorted(os.listdir(directory)):
        folder_path = os.path.join(directory, folder)
        if not os.path.isdir(folder_path):
            continue

        seed = extract_seed(folder)
        if seed is None:
            continue

        # Initialize seed data if not already present
        if seed not in seed_success:
            seed_success[seed] = []
            seed_counts[seed] = 0

        # Iterate through .npz files in the folder
        for file in sorted(os.listdir(folder_path)):
            if file.endswith("_info.npz"):
                file_path = os.path.join(folder_path, file)
                try:
                    data = np.load(file_path)
                    if "success" in data and data["success"].item() >= 0:
                        seed_success[seed].append(data["success"].item())
                        seed_counts[seed] += 1

                        env_name = file_path.split("/")[-5]
                        layout = file_path.split("/")[-2].split("_")[0][-1]
                        style = file_path.split("/")[-2].split("_")[1][-1]
                        ep_idx = file_path.split("/")[-1].split("_")[0][-1]
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")

    # Compute average success rate and print details
    all_averages = []
    for seed, successes in seed_success.items():
        if successes:
            avg_success = np.mean(successes)
            all_averages.append(avg_success)
            env_name = directory.split("/")[-3]
            method = directory.split("/")[-2]
            print(f"{env_name}, {method}, Seed {seed}: {len(successes)} runs, Average Success = {avg_success:.2f}")

    # Compute overall mean and standard deviation
    if all_averages:
        overall_mean = np.mean(all_averages)
        overall_std = np.std(all_averages)
        print(f"Overall: Mean = {overall_mean:.2f}, Std = {overall_std:.2f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute average success rate for each seed.")
    parser.add_argument("directory", type=str, help="Path to the directory containing evaluation logs.")
    args = parser.parse_args()

    compute_success_rate(args.directory)

