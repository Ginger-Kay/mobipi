"""
Makes a plot for real robot results.

Usage:
python plot_real_table.py

@yjy0625
"""
import numpy as np
import matplotlib.pyplot as plt

# Steps for each task
steps_per_task = {
    "Pick Chips": ["Collision-free", "Nav", "Grasp", "Success"],
    "Grab Paper Towels": ["Collision-free", "Nav", "Grab", "Success"],
    "Pour Tomatoes": ["Collision-free", "Nav", "Grasp", "Pour", "Success"]
}

# Methods
methods = ["Humans (not policy-aware)", "BC w/ Nav (policy-aware)", "Ours"]
colors = ["#003f5c", "#D8B9FF", "#FFA556"]

# Data (success ratios)
data = {
    "Pick Chips": np.array([
        [1 - 2/10, 8/10, 2/10, 2/10],  # Humans
        [1 - 4/10, 3/10, 0/10, 0/10],  # BC w/ Nav
        [1 - 0/10, 10/10, 7/10, 7/10]  # Ours
    ]),
    "Grab Paper Towels": np.array([
        [1 - 0/10, 10/10, 4/10, 4/10],  # Humans
        [1 - 7/10, 3/10, 3/10, 3/10],   # BC w/ Nav
        [1 - 0/10, 10/10, 10/10, 10/10] # Ours
    ]),
    "Pour Tomatoes": np.array([
        [1 - 0/10, 8/10, 2/10, 2/10, 2/10],  # Humans
        [1 - 2/10, 8/10, 7/10, 7/10, 7/10],  # BC w/ Nav
        [1 - 0/10, 10/10, 9/10, 9/10, 8/10]  # Ours
    ])
}

# Start plotting
fig, axes = plt.subplots(1, len(data), figsize=(15, 3))

for idx, (task, step_names) in enumerate(steps_per_task.items()):
    ax = axes[idx]
    num_steps = len(step_names)
    x = np.arange(num_steps)

    width = 0.25
    offset = np.linspace(-width, width, len(methods))

    # Shade the "Fatal" step background
    ax.axvspan(-0.5, 0.5, color="#FFA555", alpha=0.2)

    for m_idx, method in enumerate(methods):
        method_data = data[task][m_idx]
        bars = ax.bar(x + offset[m_idx], np.maximum(method_data, 0.02), width=width, color=colors[m_idx])

        # Only annotate Success bars
        for step_idx, (xi, yi) in enumerate(zip(x + offset[m_idx], method_data)):
            ax.text(xi, yi + 0.03, f"{int(yi * 10)}", ha='center', va='bottom', fontsize=10)

    ax.set_title(task, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(step_names, rotation=0)
    ax.set_xlim(-width * 2, len(data[task][0]) - 1 + width * 2)
    ax.set_ylim(0, 1.2)

    # y-axis ticks show as x/10
    if idx == 0:
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_yticklabels([f"{int(t*10)}/10" for t in np.linspace(0, 1, 6)])
    else:
        ax.set_yticks([])

    # Remove grid
    ax.grid(False)

    # Add border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.)

    if idx == 0:
        ax.set_ylabel("Performance (x/10)")

# Add legend
handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
fig.legend(handles, methods, loc='lower center', ncol=len(methods), fontsize=12)

plt.tight_layout(rect=[0, 0.1, 1, 1.02])
plt.savefig("real_results.pdf")
