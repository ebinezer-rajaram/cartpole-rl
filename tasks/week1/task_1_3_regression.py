import numpy as np
import os

from cartpole.plotting import (
    _make_output_dirs,
    plot_predicted_vs_true_deltas,
    plot_all_deltas_vs_inputs
)

def main():
    X = np.load("data/task_1.3/X.npy")  
    Y = np.load("data/task_1.3/Y.npy")  

    labels = ["x", "x_dot", "theta", "theta_dot"]

    C = np.linalg.lstsq(X, Y, rcond=None)[0].T 
    print("Fitted C matrix:\n", C)

    Y_pred = X @ C.T  

    out_dir = _make_output_dirs("figures/task_1.3/regression")
    plot_predicted_vs_true_deltas(Y, Y_pred, labels, out_dir)
    plot_all_deltas_vs_inputs(X, Y, Y_pred, labels, out_dir)

    mse_per_dim = np.mean((Y - Y_pred)**2, axis=0)
    print("\nMean Squared Error per output dimension:")
    for j, mse in enumerate(mse_per_dim):
        print(f"Δ{labels[j]}: {mse:.6f}")

    total_mse = np.mean((Y - Y_pred)**2)
    print(f"\nTotal MSE across all outputs: {total_mse:.6f}")

if __name__ == "__main__":
    main()
