import numpy as np
from cartpole.kernels import fit_kernel_model, predict_kernel_model
from cartpole.plotting import (
    _make_output_dirs,
    plot_predicted_vs_true_deltas,
    plot_all_deltas_vs_inputs
)


def main():
    X = np.load("data/state_transitions/X.npy") 
    Y = np.load("data/state_transitions/Y.npy")
    N = X.shape[0]
    M = 100 

    rng = np.random.default_rng(seed=0)
    indices = rng.choice(N, M, replace=False)
    basis_X = X[indices]

    lengthscales = np.std(X, axis=0)

    alpha = fit_kernel_model(X, Y, basis_X, lengthscales, lam=1e-4)

    Y_pred = predict_kernel_model(X, basis_X, alpha, lengthscales)

    labels = ["x", "x_dot", "theta", "theta_dot"]
    out_dir = _make_output_dirs("figures/kernel_model/regression")
    plot_predicted_vs_true_deltas(Y, Y_pred, labels, out_dir)
    plot_all_deltas_vs_inputs(X, Y, Y_pred, labels, out_dir)
    
    mse_per_dim = np.mean((Y - Y_pred)**2, axis=0)
    print("Mean Squared Error per output dimension:")
    for j, mse in enumerate(mse_per_dim):
        print(f"Δ{labels[j]}: {mse:.6f}")

    total_mse = np.mean((Y - Y_pred)**2)
    print(f"\nTotal MSE across all outputs: {total_mse:.6f}")

if __name__ == "__main__":
    main()
