import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad
from scipy.optimize import minimize
import os
from sklearn.cluster import KMeans

from cartpole.plotting import (
    _make_output_dirs,
    plot_predicted_vs_true_deltas,
    plot_all_deltas_vs_inputs
)

def to_sincos_features_with_action(X):
    X_new = np.zeros((X.shape[0], 6))
    X_new[:, 0] = X[:, 0]  # x
    X_new[:, 1] = X[:, 1]  # x_dot
    X_new[:, 2] = np.sin(X[:, 2])  # sin(theta)
    X_new[:, 3] = np.cos(X[:, 2])  # cos(theta)
    X_new[:, 4] = X[:, 3]  # theta_dot
    X_new[:, 5] = X[:, 4]  # action
    return X_new

def rbf_kernel_jax(X1, X2, lengthscales):
    def kernel_fn(x, y):
        scaled = (x - y) / lengthscales
        return jnp.exp(-0.5 * jnp.sum(scaled**2))
    return jax.vmap(lambda x: jax.vmap(lambda y: kernel_fn(x, y))(X2))(X1)

@jit
def predict_kernel(X_test, X_basis, alpha, lengthscales):
    K = rbf_kernel_jax(X_test, X_basis, lengthscales)
    return K @ alpha

def fit_model_jax(X_train, Y_train, X_basis, log_params):
    log_lengthscales = log_params[:6]
    log_lambda = log_params[6]

    lengthscales = jnp.exp(log_lengthscales)
    lam = jnp.exp(log_lambda)

    K = rbf_kernel_jax(X_train, X_basis, lengthscales)
    alpha = jnp.linalg.solve(K.T @ K + lam * jnp.eye(K.shape[1]), K.T @ Y_train)
    return alpha, lengthscales, lam

def loss_fn(log_params, X_train, Y_train, X_val, Y_val, X_basis):
    alpha, lengthscales, lam = fit_model_jax(X_train, Y_train, X_basis, log_params)
    Y_pred = predict_kernel(X_val, X_basis, alpha, lengthscales)
    return jnp.mean((Y_val - Y_pred) ** 2)

def main():
    labels = ["x", "x_dot", "sin(θ)", "cos(θ)", "theta_dot", "action"]
    output_labels = ["x", "x_dot", "theta", "theta_dot"]
    out_dir = _make_output_dirs("figures/task_3.1/optimization")
    
    X_raw = np.load("data/task_3.1/X.npy")
    Y = np.load("data/task_3.1/Y.npy")
    X = to_sincos_features_with_action(X_raw)

    split = 800
    X_train, Y_train = X[:split], Y[:split]
    X_val, Y_val = X[split:], Y[split:]

    # rng = np.random.default_rng(0)
    # basis_idx = rng.choice(split, size=100, replace=False)
    # X_basis = X_train[basis_idx]

    M = 200  # or 500, 1000
    kmeans = KMeans(n_clusters=M, n_init=10, random_state=0)
    kmeans.fit(X_train)
    X_basis = kmeans.cluster_centers_


    init_log_lengthscales = np.log(np.std(X_train, axis=0))
    init_log_lambda = np.log(1e-4)
    init_params = np.concatenate([init_log_lengthscales, [init_log_lambda]])

    alpha_init, _, _ = fit_model_jax(X_train, Y_train, X_basis, init_params)
    Y_val_pred_init = predict_kernel(X_val, X_basis, alpha_init, jnp.exp(init_params[:6]))
    mse_init = np.mean((Y_val - np.array(Y_val_pred_init))**2)
    print("\nInitial validation MSE:", mse_init)
    
    # Calculate per-variable initial MSE
    mse_per_var_init = np.mean((Y_val - np.array(Y_val_pred_init))**2, axis=0)
    print("\nInitial MSE per output variable:")
    for i, label in enumerate(output_labels):
        print(f"Δ{label}: {mse_per_var_init[i]:.6f}")

    objective = lambda p: float(loss_fn(p, X_train, Y_train, X_val, Y_val, X_basis))
    grad_fn = jax.grad(loss_fn)

    result = minimize(
        objective,
        init_params,
        method="L-BFGS-B",
        jac=lambda p: np.array(grad_fn(p, X_train, Y_train, X_val, Y_val, X_basis)),
        options={"maxiter": 100}
    )

    print("\nOptimization success:", result.success)
    print("Final validation MSE:", result.fun)

    opt_log_params = result.x
    opt_lengthscales = np.exp(opt_log_params[:6])
    opt_lambda = np.exp(opt_log_params[6])

    print("\nOptimal lengthscales:", opt_lengthscales)
    print("Optimal lambda:", opt_lambda)

    alpha, _, _ = fit_model_jax(X_train, Y_train, X_basis, opt_log_params)
    Y_val_pred = predict_kernel(X_val, X_basis, alpha, jnp.exp(opt_log_params[:6]))

    # Calculate per-variable final MSE
    mse_per_var = np.mean((Y_val - np.array(Y_val_pred))**2, axis=0)
    print("\nFinal MSE per output variable:")
    for i, label in enumerate(output_labels):
        print(f"Δ{label}: {mse_per_var[i]:.6f}")
    
    # Calculate percentage improvement per variable
    pct_improvement = 100 * (mse_per_var_init - mse_per_var) / mse_per_var_init
    print("\nPercentage improvement per variable:")
    for i, label in enumerate(output_labels):
        print(f"Δ{label}: {pct_improvement[i]:.2f}%")

    plot_predicted_vs_true_deltas(np.array(Y_val), np.array(Y_val_pred), output_labels, out_dir)
    plot_all_deltas_vs_inputs(np.array(X_val), np.array(Y_val), np.array(Y_val_pred), output_labels, out_dir)

    os.makedirs("models/task_3.1", exist_ok=True)
    np.savez("models/task_3.1/model_kernel_optimized_sincos_action.npz",
             X_basis=np.array(X_basis),
             alpha=np.array(alpha),
             lengthscales=np.exp(opt_log_params[:6]),
             lam=np.exp(opt_log_params[6]))
    print("\nSaved model to models/task_3.1/model_kernel_optimized_sincos_action.npz")

if __name__ == "__main__":
    main()
