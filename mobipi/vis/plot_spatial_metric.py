"""
Plots spatial metric for the simulated tasks.

Usage:
python plot_spatial_metric.py

@yjy0625
"""
import numpy as np
import matplotlib.pyplot as plt

# Perturbation scales
scales = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# Parsed mean values
tasks = {
    "Close Door":        [0.78, 0.64, 0.49, 0.33, 0.24, 0.14, 0.16],
    "Turn on Faucet":    [0.79, 0.48, 0.31, 0.18, 0.11, 0.07, 0.03],
    "Turn on Microwave": [0.42, 0.32, 0.18, 0.10, 0.07, 0.03, 0.03],
    "Close Drawer":      [0.92, 0.93, 0.77, 0.70, 0.57, 0.50, 0.32],
    "Turn on Stove":     [0.48, 0.31, 0.12, 0.03, 0.01, 0.00, 0.03],
}

# Exponential decay fitting
def fit_exponential(x, y):
    log_y = np.log(np.clip(y, 1e-5, None))
    A = np.vstack([x, np.ones(len(x))]).T
    k, log_y0 = np.linalg.lstsq(A, log_y, rcond=None)[0]
    y0 = np.exp(log_y0)
    return y0, -k

# Plotting
fig, axes = plt.subplots(1, len(tasks), figsize=(18, 4), sharey=True)

for i, (task_name, success_rates) in enumerate(tasks.items()):
    x = np.array(scales)
    y = np.array(success_rates)

    # Fit curve
    y0, k = fit_exponential(x, y)
    x_continuous = np.linspace(-0.1, 0.4, 300)
    fitted = y0 * np.exp(-k * x_continuous)
    phi = np.log(2) / k
    y_half = y0 / 2

    # Plot curve and data
    ax = axes[i]
    ax.plot(x, y, 'o', color="#3498DB", markersize=8, label="Policy Performance")
    ax.plot(x_continuous, fitted, '-', color="#1ABC9C", linewidth=2, label="Fitted Curve")

    # Vertical line at 0 and phi
    ax.vlines(phi, ymin=-0.05, ymax=y_half, linestyle='--', color="#FFA555", linewidth=1)
    ax.vlines(0, ymin=-0.05, ymax=y0, linestyle='--', color="#FFA555", linewidth=1)

    # Mark fitted curve at S_\pi(0) and S_\pi(\phi)
    ax.plot(phi, y_half, marker=r'$\odot$', color="#FFA555", markersize=12)
    ax.plot(phi, y_half, marker="o", color="#FFA555", markersize=5)
    ax.plot(0, y0, 'o', color="#FFA555", markersize=6)

    # Horizontal lines from y axis to a bit right of phi
    x_end = phi + (0.03 if i < 4 else 0.05)
    ax.hlines(y0, xmin=-0.025, xmax=x_end, linestyle='-', color="#FFA555", linewidth=1)
    ax.hlines(y_half, xmin=-0.025, xmax=x_end, linestyle='-', color="#FFA555", linewidth=1)

    # Arrow pointing from full to half performance
    arrow_x = x_end - 0.01
    ax.annotate(
        '',
        xy=(arrow_x, y_half),
        xytext=(arrow_x, y0),
        arrowprops=dict(arrowstyle="->", color="#FFA555", linewidth=2,
                        mutation_scale=20),
    )

    # Text next to arrow for leftmost plot
    if i == 0:
        ax.text(arrow_x + 0.01, (y0 + y_half) / 2 + 0.06,
                r"performance",
                va='center', ha='left', fontsize=14, color="#FFA555", fontweight="bold")
        ax.text(arrow_x + 0.01, (y0 + y_half) / 2 - 0.06,
                r"drops by 50%",
                va='center', ha='left', fontsize=14, color="#FFA555", fontweight="bold")

    # Axis labels and titles
    ax.set_title(task_name, fontsize=14)
    ax.set_xlabel("Perturbation Scale", fontsize=14)
    if i == 0:
        ax.set_ylabel("Success Rate", fontsize=14)

    if i == 0:
        ax.legend(fontsize=10, loc="upper right")

    # Phi label
    ax.text(0.5, -0.3, f"$\\phi$ = {phi:.3f}", transform=ax.transAxes,
            fontsize=14, fontweight="bold", ha='center', color="#FFA555")

    ax.set_xlim(-0.025, 0.325)
    ax.set_ylim(-0.05, 1.1)

plt.tight_layout()
plt.savefig("spatial_metric.pdf")
