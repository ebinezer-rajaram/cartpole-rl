import numpy as np
import matplotlib.pyplot as plt
import os
from cartpole.scanning import perform_single_step
from cartpole.CartPole import CartPole

def main():
    labels = ["x", "x_dot", "theta", "theta_dot"]
    scan_ranges = {
        0: np.linspace(-5, 5, 100),
        1: np.linspace(-10, 10, 100),
        2: np.linspace(-np.pi, np.pi, 100),
        3: np.linspace(-15, 15, 100),
    }

    # Load trained model matrix C (4 × d)
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

    for i in range(4):  # scan each input dim
        scan_vals = scan_ranges[i]
        Y_true = []
        Y_pred = []

        for val in scan_vals:
            state = base_state.copy()
            state[i] = val

            y = perform_single_step(state, return_delta=True)
            y_hat = C @ state[:d]  # predicted Δ

            Y_true.append(y)
            Y_pred.append(y_hat)

        Y_true = np.array(Y_true)
        Y_pred = np.array(Y_pred)

        for j in range(4):  # plot Δ_j vs X_i
            plt.figure()
            plt.plot(scan_vals, Y_true[:, j], label="true", alpha=0.6)
            plt.plot(scan_vals, Y_pred[:, j], '--', label="pred", alpha=0.6)
            plt.xlabel(labels[i])
            plt.ylabel(f"Δ{labels[j]}")
            plt.title(f"Δ{labels[j]} vs {labels[i]} (true vs pred)")
            plt.legend()
            plt.grid(True)

            fname = f"delta_{labels[j]}_vs_{labels[i]}"
            plt.savefig(f"{fig_base}/pdf/{fname}.pdf")
            plt.savefig(f"{fig_base}/png/{fname}.png")
            plt.close()

if __name__ == "__main__":
    main()
