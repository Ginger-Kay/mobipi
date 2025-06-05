"""
Makes a plot for sim experiment results.

Usage:
python plot_sim_table.py

@yjy0625
"""
import numpy as np
import matplotlib.pyplot as plt

# Tasks
tasks = ["Close Door", "Turn on Faucet", "Turn on Microwave", "Close Drawer", "Turn on Stove"]

# Methods
methods = [
    "Manipulation Policy",
    "Manipulation Policy (perturb 5 cm)",
    "Manipulation Policy (perturb 10 cm)",
    "LeLaN (non-policy-aware)",
    "VLFM (non-policy-aware)",
    "BC w/ Nav (policy-aware)",
    "Ours"
]

# Data
means = np.array([
    [0.78, 0.79, 0.42, 0.92, 0.48],  # Manipulation Policy
    [0.64, 0.48, 0.32, 0.93, 0.31],  # Manipulation Policy (5cm)
    [0.49, 0.31, 0.18, 0.77, 0.12],  # Manipulation Policy (10cm)
    [0.12, 0.05, 0.03, 0.15, 0.01],  # LeLaN
    [0.15, 0.00, 0.06, 0.17, 0.03],  # VLFM
    [0.23, 0.03, 0.05, 0.11, 0.01],  # BC w/ Nav
    [0.62, 0.50, 0.37, 0.29, 0.33],  # Ours
])

stds = np.array([
    [0.03, 0.04, 0.07, 0.03, 0.06],
    [0.03, 0.02, 0.06, 0.02, 0.08],
    [0.01, 0.02, 0.02, 0.02, 0.04],
    [0.03, 0.04, 0.01, 0.05, 0.01],
    [0.07, 0.00, 0.03, 0.04, 0.01],
    [0.15, 0.05, 0.02, 0.04, 0.02],
    [0.00, 0.08, 0.06, 0.04, 0.03],
])

# Colors (starting from perturb 5cm onwards)
colors = [
    "#BDC3C7",  # Perturb 5 cm
    "#95A5A6",  # Perturb 10 cm
    "#1ABC9C",  # LeLaN
    "#3498DB",  # VLFM
    '#D2B8F2',  # BC w/ Nav
    "#FFA556",  # Ours
]

# Create figure
fig, axes = plt.subplots(1, len(tasks), figsize=(15, 4), sharey=True)

for i, task in enumerate(tasks):
    ax = axes[i]

    # Background
    ax.axvspan(-0.7, 1.5, color="#ECF0F1", alpha=1.0)

    # Best possible performance
    best_perf = np.max(means[0:3, i])

    # Plot dashed line for best possible performance
    ax.axhline(best_perf, color="#333", linestyle="--", linewidth=1.5)

    # Add text saying best possible performance
    if i == 0:
        ax.text(
            5.5, best_perf + 0.04,  # x position near the right, 5.8 out of 6 (adjustable)
            "best possible performance ▼",
            ha="right",  # right-align the text
            va="bottom",
            color="#333",
            fontsize=8
        )

    # Prepare bars: skip the first method (index 0)
    selected_means = means[1:, i]
    selected_stds = stds[1:, i]

    ax.bar(
        np.arange(len(selected_means)),
        selected_means,
        yerr=selected_stds,
        color=colors,
        capsize=5
    )

    ax.set_title(task, pad=10)
    ax.set_xticks([])
    ax.set_xlim(-0.7, 5.7)
    ax.set_ylim(0, 1)

# Create legend
handles = [
    plt.Line2D([0], [0], color="#333", linestyle="--", linewidth=1.5)
] + [
    plt.Rectangle((0, 0), 1, 1, color=color) for color in colors
]
labels = [
    "Manipulation Policy",
    "Manipulation Policy (perturb 5 cm)",
    "Manipulation Policy (perturb 10 cm)",
    "LeLaN (not policy-aware)",
    "VLFM (not policy-aware)",
    "BC w/ Nav (policy-aware)",
    "Ours",
]

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    fontsize=11,
    bbox_to_anchor=(0.5, 0.03),
)

plt.tight_layout(rect=[0, 0.14, 1, 1.01])
plt.savefig("sim_results.pdf")
