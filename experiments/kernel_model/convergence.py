import numpy as np
import matplotlib.pyplot as plt
import os

from cartpole.data import collect_dataset
from cartpole.kernels import fit_kernel_model, predict_kernel_model
from cartpole.plotting import _make_output_dirs

def evaluate_model(X_train, Y_train, X_val, Y_val, M, lam=1e-4):
    rng = np.random.default_rng(seed=42)
    idx = rng.choice(len(X_train), size=M, replace=False)
    basis_X = X_train[idx]
    lengthscales = np.std(X_train, axis=0)

    alpha = fit_kernel_model(X_train, Y_train, basis_X, lengthscales, lam)
    Y_pred = predict_kernel_model(X_val, basis_X, alpha, lengthscales)
    return np.mean((Y_val - Y_pred) ** 2)

def main():
    X_test, Y_test = collect_dataset(500)

    # M sweep (N fixed)
    N_fixed = 500
    M_vals = [10, 20, 40, 80, 160]
    X_train, Y_train = collect_dataset(N_fixed)

    mse_M = []
    for M in M_vals:
        print(f"Evaluating M = {M}")
        mse_M.append(evaluate_model(X_train, Y_train, X_test, Y_test, M))

    # N sweep (M fixed)
    M_fixed = 40
    N_vals = [100, 200, 400, 800, 1600]
    mse_N = []

    for N in N_vals:
        print(f"Evaluating N = {N}")
        X_train, Y_train = collect_dataset(N)
        mse_N.append(evaluate_model(X_train, Y_train, X_test, Y_test, M_fixed))

    out_dir = _make_output_dirs("figures/kernel_model/convergence")

    plt.figure()
    plt.plot(M_vals, mse_M, 'o-')
    plt.xlabel("Basis Count M")
    plt.ylabel("MSE")
    plt.title("Convergence: Varying M (N = 500)")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir["pdf"], "mse_vs_M_N_fixed.pdf"))
    plt.savefig(os.path.join(out_dir["png"], "mse_vs_M_N_fixed.png"))
    plt.close()

    plt.figure()
    plt.plot(N_vals, mse_N, 'o-')
    plt.xlabel("Training Points N")
    plt.ylabel("MSE")
    plt.title("Convergence: Varying N (M = 40)")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir["pdf"], "mse_vs_N_M_fixed.pdf"))
    plt.savefig(os.path.join(out_dir["png"], "mse_vs_N_M_fixed.png"))
    plt.close()

    print("\nMSE vs M (N = 500):")
    for m, e in zip(M_vals, mse_M):
        print(f"  M = {m:<4} → MSE = {e:.6f}")

    print("\nMSE vs N (M = 40):")
    for n, e in zip(N_vals, mse_N):
        print(f"  N = {n:<4} → MSE = {e:.6f}")

if __name__ == "__main__":
    main()
