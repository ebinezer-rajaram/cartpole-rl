# task_2_1_nonlinear_regression.py

import numpy as np
import matplotlib.pyplot as plt
from cartpole.plotting import _make_output_dirs, plot_predicted_vs_true_deltas, plot_deltas_vs_inputs
from cartpole.kernels import periodic_kernel

def fit_kernel_model(X, Y, basis_X, lengthscales, lam=1e-4):
    K = periodic_kernel(X, basis_X, lengthscales)  # (N, M)
    alpha = np.linalg.solve(K.T @ K + lam * np.eye(K.shape[1]), K.T @ Y)  # (M, 4)
    return alpha

def predict_kernel_model(X_new, basis_X, alpha, lengthscales):
    K_new = periodic_kernel(X_new, basis_X, lengthscales)  # (N_new, M)
    return K_new @ alpha  # (N_new, 4)

def plot_results(X, Y, Y_pred, save_base):
    labels = ["x", "x_dot", "theta", "theta_dot"]
    dirs = _make_output_dirs(save_base)
    plot_predicted_vs_true_deltas(Y, Y_pred, labels, dirs)
    plot_deltas_vs_inputs(X, Y, Y_pred, labels, dirs)
    


def main():
    X = np.load("data/task_1.3/X.npy")  # (N, 4)
    Y = np.load("data/task_1.3/Y.npy")  # (N, 4)
    N = X.shape[0]
    M = 100  

    rng = np.random.default_rng(seed=0)
    indices = rng.choice(N, M, replace=False)
    basis_X = X[indices]

    lengthscales = np.std(X, axis=0)

    alpha = fit_kernel_model(X, Y, basis_X, lengthscales, lam=1e-4)

    Y_pred = predict_kernel_model(X, basis_X, alpha, lengthscales)

    save_base = "figures/task_2.1"
    plot_results(X, Y, Y_pred, save_base)

    mse_per_dim = np.mean((Y - Y_pred)**2, axis=0)
    print("MSE per output dimension:")
    for i, mse in enumerate(mse_per_dim):
        print(f"Δ{['x','x_dot','theta','theta_dot'][i]}: {mse:.6f}")
    print(f"Total MSE: {np.mean((Y - Y_pred)**2):.6f}")

if __name__ == "__main__":
    main()
