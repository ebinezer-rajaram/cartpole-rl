import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from cartpole.jax_dynamics import rollout_jax
from cartpole.plotting import plot_time_series, _make_output_dirs

def trajectory_loss_jax(traj, X0=None, sigma_l=0.5):
    if X0 is None:
        X0 = jnp.zeros(4)
    sigma_l = jnp.asarray(sigma_l)
    diffs = (traj - X0) / sigma_l
    sq_norms = jnp.sum(diffs**2, axis=-1)
    losses = 1 - jnp.exp(-0.5 * sq_norms)
    return jnp.sum(losses)

def objective(policy, x0, T, params, sigma_l):
    traj = rollout_jax(x0, policy, T, params)
    return trajectory_loss_jax(traj, X0=jnp.zeros(4), sigma_l=sigma_l)

def optimise_and_plot(x0, T, params, sigma_l, out_dir, tag):
    policy0 = np.zeros(4)
    obj_and_grad = jax.jit(jax.value_and_grad(lambda p: objective(p, x0, T, params, sigma_l)))
    res = minimize(
        lambda p: float(obj_and_grad(p)[0]),
        policy0,
        method="L-BFGS-B",
        jac=lambda p: np.array(obj_and_grad(p)[1]),
        options={"maxiter": 100}
    )
    print(f"=== {tag.upper()} ===")
    print("Initial state:", np.array(x0))
    print("Sigma_l:", np.array(sigma_l))
    print("Optimized policy:", res.x)
    print("Final loss:", res.fun)
    traj = np.array(rollout_jax(x0, res.x, T, params))
    plot_time_series(traj, f"optimised_policy_jax_{tag}", out_dir)
    return res, traj

def main():
    out_dir = "figures/task_3.2/optimise_policy_time_series"

    params = dict(
        pole_length=0.5,
        pole_mass=0.5,
        cart_mass=0.5,
        mu_c=0.001,
        mu_p=0.001,
        gravity=9.8,
        sim_steps=50,
        delta_time=0.1,
        max_force=40.0,
    )
    T = 20

    sigma_l_list = [
        jnp.array([0.5, 0.4, 0.05, 0.25]),
        jnp.array([0.3, 0.3, 0.05, 0.2]),
        # Add more as needed
    ]

    x0_list = [
        ("good_upright", jnp.array([0.0, 0.0, 0.1, 0.0])),
        ("bad_downward", jnp.array([0.0, 0.0, jnp.pi, 0.0])),
        ("random_realistic", jnp.array([0.08, -0.13, 0.09, 0.17]))
        # Add more as needed
    ]

    for sigma_idx, sigma_l in enumerate(sigma_l_list):
        for x0_tag, x0 in x0_list:
            tag = f"{x0_tag}_sig{sigma_idx}"
            optimise_and_plot(x0, T, params, sigma_l, out_dir, tag=tag)

if __name__ == "__main__":
    main()
