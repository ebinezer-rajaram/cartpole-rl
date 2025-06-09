import numpy as np
import matplotlib.pyplot as plt
import os
from cartpole.scanning import perform_single_step

# Set global plotting parameters for better readability in reports
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15, 
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 13,
    'figure.titlesize': 16
})

def main():
    labels = ["x", "x_dot", "theta", "theta_dot"]
    scan_ranges = {
        0: np.linspace(-5, 5, 100),
        1: np.linspace(-10, 10, 100),
        2: np.linspace(-np.pi, np.pi, 100),
        3: np.linspace(-15, 15, 100),
    }

    X_data = np.load("data/task_1.3/X.npy")
    Y_data = np.load("data/task_1.3/Y.npy")
    C = np.linalg.lstsq(X_data, Y_data, rcond=None)[0].T

    base_state = np.random.uniform(
        low=[-5, -10, -np.pi, -15],
        high=[5, 10, np.pi, 15]
    )

    d = X_data.shape[1]
    fig_base = "figures/task_1.3/scan_compare"
    os.makedirs(os.path.join(fig_base, "pdf"), exist_ok=True)
    os.makedirs(os.path.join(fig_base, "png"), exist_ok=True)

    for i in range(4):  
        scan_vals = scan_ranges[i]
        Y_true = []
        Y_pred = []

        for val in scan_vals:
            state = base_state.copy()
            state[i] = val

            y = perform_single_step(state, return_delta=True)
            y_hat = C @ state[:d] 

            Y_true.append(y)
            Y_pred.append(y_hat)

        Y_true = np.array(Y_true)
        Y_pred = np.array(Y_pred)

        # Individual Δj vs Xi plots
        for j in range(4):
            plt.figure(figsize=(6, 5))
            plt.plot(scan_vals, Y_true[:, j], label="true", alpha=0.6, linewidth=2.0)
            plt.plot(scan_vals, Y_pred[:, j], '--', label="pred", alpha=0.6, linewidth=2.0)
            plt.xlabel(labels[i])
            plt.ylabel(f"Δ{labels[j]}")
            plt.title(f"Δ{labels[j]} vs {labels[i]} (true vs pred)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            fname = f"delta_{labels[j]}_vs_{labels[i]}"
            plt.savefig(f"{fig_base}/pdf/{fname}.pdf", bbox_inches='tight')
            plt.savefig(f"{fig_base}/png/{fname}.png", bbox_inches='tight')
            plt.close()

        # Combined plot: all Δj vs Xi
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axs = axs.flatten()

        for j in range(4):
            axs[j].plot(scan_vals, Y_true[:, j], label="true", alpha=0.6, linewidth=2.0)
            axs[j].plot(scan_vals, Y_pred[:, j], '--', label="pred", alpha=0.6, linewidth=2.0)
            axs[j].set_ylabel(f"Δ{labels[j]}")
            axs[j].set_title(f"Δ{labels[j]} vs {labels[i]}")
            axs[j].grid(True)
            axs[j].legend()

        axs[-1].set_xlabel(labels[i])
        fig.suptitle(f"All Δs vs {labels[i]} (true vs predicted)")
        plt.subplots_adjust(wspace=0.15, hspace=0.2, top=0.9)

        fname = f"combined_deltas_vs_{labels[i]}"
        fig.savefig(f"{fig_base}/pdf/{fname}.pdf", bbox_inches='tight')
        fig.savefig(f"{fig_base}/png/{fname}.png", bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    main()
