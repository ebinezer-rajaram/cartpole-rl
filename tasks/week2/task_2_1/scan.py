import numpy as np
import os

from cartpole.kernels import fit_kernel_model, predict_kernel_model
from cartpole.plotting import _make_output_dirs, plot_scan_comparison
from cartpole.scanning import scan_model_vs_true

def main():
    labels = ["x", "x_dot", "theta", "theta_dot"]
    scan_ranges = {
        0: np.linspace(-5, 5, 100),
        1: np.linspace(-10, 10, 100),
        2: np.linspace(-np.pi, np.pi, 100),
        3: np.linspace(-15, 15, 100),
    }

    X = np.load("data/task_1.3/X.npy")
    Y = np.load("data/task_1.3/Y.npy")
    N, M = X.shape[0], 100

    rng = np.random.default_rng(seed=0)
    basis_X = X[rng.choice(N, M, replace=False)]
    lengthscales = np.std(X, axis=0)
    alpha = fit_kernel_model(X, Y, basis_X, lengthscales)

    model_predict_fn = lambda x: predict_kernel_model(x[None, :], basis_X, alpha, lengthscales)[0]

    base_state = np.random.uniform(
        low=[-5, -10, -np.pi, -15],
        high=[5, 10, np.pi, 15]
    )

    out_dir = _make_output_dirs("figures/task_2.1/scan")

    for i in range(4):
        scan_vals, Y_true, Y_pred = scan_model_vs_true(i, scan_ranges[i], base_state, model_predict_fn)
        plot_scan_comparison(scan_vals, Y_true, Y_pred, labels[i], labels, out_dir, tag="model")

if __name__ == "__main__":
    main()
