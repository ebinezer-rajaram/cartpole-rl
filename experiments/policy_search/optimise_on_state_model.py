import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from cartpole.plotting import plot_time_series, _make_output_dirs

# ----- Kernel Regression: JAX version for 5-feature model -----
def rbf_kernel_jax(X1, X2, lengthscales):
    def kernel_fn(x, y):
        scaled = (x - y) / lengthscales
        return jnp.exp(-0.5 * jnp.sum(scaled**2))
    return jax.vmap(lambda x: jax.vmap(lambda y: kernel_fn(x, y))(X2))(X1)

@jax.jit
def predict_kernel(X_test, X_basis, alpha, lengthscales):
    K = rbf_kernel_jax(X_test, X_basis, lengthscales)
    return K @ alpha

def build_features(x):
    return jnp.array([
        x[0],
        x[1],
        jnp.sin(x[2]),
        jnp.cos(x[2]),
        x[3]
    ])[None, :]

# ----- Model rollout using learned dynamics (sin/cos kernel model, state only) -----
def model_rollout_jax(x0, policy, T, X_basis, alpha, lengthscales):
    def step(x, _):
        a = jnp.dot(policy, x)
        features = build_features(x)
        dx = predict_kernel(features, X_basis, alpha, lengthscales)[0]
        x_next = x + dx
        # Remap angle
        x_next = x_next.at[2].set(((x_next[2] + jnp.pi) % (2 * jnp.pi)) - jnp.pi)
        return x_next, x_next
    _, traj = jax.lax.scan(step, x0, None, length=T)
    return traj

def trajectory_loss_jax(traj, X0=None, sigma_l=0.5):
    if X0 is None:
        X0 = jnp.zeros(4)
    sigma_l = jnp.asarray(sigma_l)
    diffs = (traj - X0) / sigma_l
    sq_norms = jnp.sum(diffs**2, axis=-1)
    losses = 1 - jnp.exp(-0.5 * sq_norms)
    return jnp.sum(losses)

def main():
    out_dir = "figures/policy_search/learned_model_policy"
    T = 13
    sigma_l = jnp.array([0.5, 0.4, 0.05, 0.25])
    x0 = jnp.array([ 0.08, -0.13,  0.09,  0.17])
    p0 = np.zeros(4)

    # Load the sin/cos kernel model
    model = np.load("models/kernel_model/model_kernel_optimized_sincos.npz")
    X_basis = jnp.array(model["X_basis"])
    alpha = jnp.array(model["alpha"])
    lengthscales = jnp.array(model["lengthscales"])

    def objective(p):
        traj = model_rollout_jax(x0, p, T, X_basis, alpha, lengthscales)
        return trajectory_loss_jax(traj, X0=jnp.zeros(4), sigma_l=sigma_l)

    obj_and_grad = jax.jit(jax.value_and_grad(objective))

    res = minimize(
        lambda p: float(obj_and_grad(p)[0]),
        p0,
        jac=lambda p: np.array(obj_and_grad(p)[1]),
        method="L-BFGS-B",
        options={"maxiter": 100}
    )
    print("Optimized policy (on model):", res.x)
    print("Final model rollout loss:", res.fun)

    # Plot model rollout (predicted)
    traj_model = np.array(model_rollout_jax(x0, res.x, T, X_basis, alpha, lengthscales))
    plot_time_series(traj_model, "model_policy_model_rollout_jax", out_dir)

    # Compare with true simulator
    from cartpole.simulation import rollout
    def linear_policy_fn(t, state): return np.dot(res.x, state)
    traj_true = rollout(np.array(x0), T=T, action_fn=linear_policy_fn, remap=True)
    plot_time_series(traj_true, "model_policy_true_rollout_jax", out_dir)

if __name__ == "__main__":
    main()
