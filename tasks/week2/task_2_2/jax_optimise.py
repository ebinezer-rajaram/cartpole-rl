import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad
from scipy.optimize import minimize
import os

from cartpole.data import collect_dataset
from cartpole.kernels import periodic_kernel  # for reference
from cartpole.plotting import _make_output_dirs, plot_predicted_vs_true_deltas, plot_all_deltas_vs_inputs

def periodic_kernel_jax(X1, X2, lengthscales, theta_index=2):
    def kernel_fn(x, y):
        diff = x - y
        diff = diff.at[theta_index].set(jnp.sin((x[theta_index] - y[theta_index]) / 2.0))
        scaled = diff / lengthscales
        return jnp.exp(-0.5 * jnp.sum(scaled ** 2))

    return jax.vmap(lambda x: jax.vmap(lambda y: kernel_fn(x, y))(X2))(X1)

@jit
def predict_kernel(X_test, X_basis, alpha, lengthscales):
    K = periodic_kernel_jax(X_test, X_basis, lengthscales)
    return K @ alpha

def fit_model_jax(X_train, Y_train, X_basis, log_params):
    log_lengthscales = log_params[:4]
    log_lambda = log_params[4]

    lengthscales = jnp.exp(log_lengthscales)
    lam = jnp.exp(log_lambda)

    K = periodic_kernel_jax(X_train, X_basis, lengthscales)
    KTK = K.T @ K
    alpha = jnp.linalg.solve(KTK + lam * jnp.eye(K.shape[1]), K.T @ Y_train)
    return alpha, lengthscales, lam

def loss_fn(log_params, X_train, Y_train, X_val, Y_val, X_basis):
    alpha, lengthscales, lam = fit_model_jax(X_train, Y_train, X_basis, log_params)
    Y_pred = predict_kernel(X_val, X_basis, alpha, lengthscales)
    return jnp.mean((Y_val - Y_pred) ** 2)

def main():
    labels = ["x", "x_dot", "theta", "theta_dot"]
    out_dir = _make_output_dirs("figures/task_2.2/optimization")

    X_all, Y_all = collect_dataset(1000)
    split = 800
    X_train, Y_train = X_all[:split], Y_all[:split]
    X_val, Y_val = X_all[split:], Y_all[split:]

    rng = np.random.default_rng(0)
    basis_idx = rng.choice(split, size=100, replace=False)
    X_basis = X_train[basis_idx]

    init_log_lengthscales = np.log(np.std(X_train, axis=0))
    init_log_lambda = np.log(1e-4)
    init_params = np.concatenate([init_log_lengthscales, [init_log_lambda]])

    print("Initial log params:", init_params)

    alpha_init, _, _ = fit_model_jax(X_train, Y_train, X_basis, init_params)
    Y_val_pred_init = predict_kernel(X_val, X_basis, alpha_init, jnp.exp(init_params[:4]))
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
    opt_lengthscales = np.exp(opt_log_params[:4])
    opt_lambda = np.exp(opt_log_params[4])

    print("\nOptimal lengthscales:", opt_lengthscales)
    print("Optimal lambda:", opt_lambda)

    alpha, _, _ = fit_model_jax(X_train, Y_train, X_basis, opt_log_params)
    Y_val_pred = predict_kernel(X_val, X_basis, alpha, jnp.exp(opt_log_params[:4]))
    
    plot_predicted_vs_true_deltas(np.array(Y_val), np.array(Y_val_pred), labels, out_dir)
    plot_all_deltas_vs_inputs(np.array(X_val), np.array(Y_val), np.array(Y_val_pred), labels, out_dir)

    mse_dim = np.mean((Y_val - np.array(Y_val_pred))**2, axis=0)
    print("\nMSE per output dim (val):")
    for i, mse in enumerate(mse_dim):
        print(f"Δ{labels[i]}: {mse:.6f}")
    print(f"Total MSE: {np.mean(mse_dim):.6f}")
    
    
    os.makedirs("models/task_2.2", exist_ok=True)
    
    np.savez("models/task_2.2/model_kernel_optimized.npz",
        X_basis=np.array(X_basis),
        alpha=np.array(alpha),
        lengthscales=np.exp(opt_log_params[:4]),
        lam=np.exp(opt_log_params[4]))
    
    print("\nSaved optimized model to models/task_2.2/model_kernel_optimized.npz")


if __name__ == "__main__":
    main()
