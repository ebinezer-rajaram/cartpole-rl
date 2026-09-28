import numpy as np
import os

from cartpole.plotting import (
    _make_output_dirs,
    plot_predicted_vs_true_deltas,
    plot_all_deltas_vs_inputs
)

def to_sincos_features(X):
    X_new = np.zeros((X.shape[0], 5))
    X_new[:, 0] = X[:, 0]              # x
    X_new[:, 1] = X[:, 1]              # x_dot
    X_new[:, 2] = np.sin(X[:, 2])      # sin(theta)
    X_new[:, 3] = np.cos(X[:, 2])      # cos(theta)
    X_new[:, 4] = X[:, 3]              # theta_dot
    return X_new

def main():
    labels_in = ["x", "x_dot", "sin(theta)", "cos(theta)", "theta_dot"]
    labels_out = ["x", "x_dot", "theta", "theta_dot"]

    # Load and transform data
    X_raw = np.load("data/state_transitions/X.npy")  # shape (N, 4)
    Y = np.load("data/state_transitions/Y.npy")      # shape (N, 4)
    X = to_sincos_features(X_raw)           # shape (N, 5)

    # Linear regression
    W = np.linalg.lstsq(X, Y, rcond=None)[0].T  # shape (4, 5)
    Y_pred = X @ W.T

    # Plotting
    out_dir = _make_output_dirs("figures/kernel_model/sincos/linear_sincos")
    plot_predicted_vs_true_deltas(Y, Y_pred, labels_out, out_dir)
    plot_all_deltas_vs_inputs(X, Y, Y_pred, labels_out, out_dir)

    # MSE report
    mse_dim = np.mean((Y - Y_pred) ** 2, axis=0)
    print("\nLinear Model (sincos) — MSE per output dimension:")
    for i, mse in enumerate(mse_dim):
        print(f"Δ{labels_out[i]}: {mse:.6f}")
    print(f"Total MSE: {np.mean(mse_dim):.6f}")

if __name__ == "__main__":
    main()

