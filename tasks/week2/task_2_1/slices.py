import numpy as np
import os
from itertools import combinations

from cartpole.kernels import fit_kernel_model, predict_kernel_model
from cartpole.scanning import scan2d_model_vs_true
from cartpole.plotting import _make_output_dirs, plot_2d_slices

def main():
    labels = ["x", "x_dot", "theta", "theta_dot"]
    scan_ranges = {
        0: (-5, 5),
        1: (-10, 10),
        2: (-np.pi, np.pi),
        3: (-15, 15),
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

    out_dir = _make_output_dirs("figures/task_2.1/slices")

    for i, j in combinations(range(4), 2):
        print(f"Scanning over ({labels[i]}, {labels[j]})")
        X_coords, Y_true, Y_pred = scan2d_model_vs_true(
            i, j,
            scan_ranges[i], scan_ranges[j],
            base_state, model_predict_fn,
            grid_resolution=30
        )

        for k in range(4):
            output_label = f"Δ{labels[k]}"
            plot_2d_slices(
                X_coords, Y_true[:, k], Y_pred[:, k],
                labels[i], labels[j], output_label,
                out_dir, tag="model"
            )

if __name__ == "__main__":
    main()
