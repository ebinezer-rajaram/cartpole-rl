import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad
from scipy.optimize import minimize
import os

from cartpole.data import collect_dataset
from cartpole.plotting import (
    _make_output_dirs,
    plot_predicted_vs_true_deltas,
    plot_all_deltas_vs_inputs
)

def to_sincos_features(X):
    X_new = np.zeros((X.shape[0], 5))
    X_new[:, 0] = X[:, 0]
    X_new[:, 1] = X[:, 1]
    X_new[:, 2] = np.sin(X[:, 2])
    X_new[:, 3] = np.cos(X[:, 2])
    X_new[:, 4] = X[:, 3]
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
    log_lengthscales = log_params[:5]
    log_lambda = log_params[5]

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
    labels = ["x", "x_dot", "sin(θ)", "cos(θ)", "theta_dot"]
    output_labels = ["x", "x_dot", "theta", "theta_dot"]
    out_dir = _make_output_dirs("figures/kernel_model/sincos/optimization")

    X_raw, Y = collect_dataset(1000)
    X = to_sincos_features(X_raw)

    split = 800
    X_train, Y_train = X[:split], Y[:split]
    X_val, Y_val = X[split:], Y[split:]
    
    rng = np.random.default_rng(0)
    basis_idx = rng.choice(split, size=100, replace=False)
    X_basis = X_train[basis_idx]

    init_log_lengthscales = np.log(np.std(X_train, axis=0))
    init_log_lambda = np.log(1e-4)
    init_params = np.concatenate([init_log_lengthscales, [init_log_lambda]])

    print("Initial log params:", init_params)

    alpha_init, _, _ = fit_model_jax(X_train, Y_train, X_basis, init_params)
    Y_val_pred_init = predict_kernel(X_val, X_basis, alpha_init, jnp.exp(init_params[:5]))
    mse_init = np.mean((Y_val - np.array(Y_val_pred_init))**2)
    print("\nInitial validation MSE:", mse_init)

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
    opt_lengthscales = np.exp(opt_log_params[:5])
    opt_lambda = np.exp(opt_log_params[5])

    print("\nOptimal lengthscales:", opt_lengthscales)
    print("Optimal lambda:", opt_lambda)

    alpha, _, _ = fit_model_jax(X_train, Y_train, X_basis, opt_log_params)
    Y_val_pred = predict_kernel(X_val, X_basis, alpha, jnp.exp(opt_log_params[:5]))

    plot_predicted_vs_true_deltas(np.array(Y_val), np.array(Y_val_pred), output_labels, out_dir)
    plot_all_deltas_vs_inputs(np.array(X_val), np.array(Y_val), np.array(Y_val_pred), output_labels, out_dir)

    mse_dim = np.mean((Y_val - np.array(Y_val_pred))**2, axis=0)
    print("\nMSE per output dim (val):")
    for i, mse in enumerate(mse_dim):
        print(f"Δ{output_labels[i]}: {mse:.6f}")
    print(f"Total MSE: {np.mean(mse_dim):.6f}")

    os.makedirs("models/kernel_model", exist_ok=True)
    np.savez("models/kernel_model/model_kernel_optimized_sincos.npz",
             X_basis=np.array(X_basis),
             alpha=np.array(alpha),
             lengthscales=np.exp(opt_log_params[:5]),
             lam=np.exp(opt_log_params[5]))
    print("\nSaved model to models/kernel_model/model_kernel_optimized_sincos.npz")

if __name__ == "__main__":
    main()
