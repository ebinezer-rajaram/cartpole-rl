import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from cartpole.plotting import _make_output_dirs
import matplotlib.pyplot as plt
import os

# Set global plotting parameters for readability in saved figures
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 13,
    'figure.titlesize': 16
})

# ----- Kernel Regression: JAX version for 6-feature model -----
def rbf_kernel_jax(X1, X2, lengthscales):
    def kernel_fn(x, y):
        scaled = (x - y) / lengthscales
        return jnp.exp(-0.5 * jnp.sum(scaled**2))
    return jax.vmap(lambda x: jax.vmap(lambda y: kernel_fn(x, y))(X2))(X1)

@jax.jit
def predict_kernel(X_test, X_basis, alpha, lengthscales):
    K = rbf_kernel_jax(X_test, X_basis, lengthscales)
    return K @ alpha

def build_features(x, a):
    return jnp.array([
        x[0],
        x[1],
        jnp.sin(x[2]),
        jnp.cos(x[2]),
        x[3],
        a
    ])[None, :]

# ----- Model rollout using learned dynamics (with force clipping) -----
def model_rollout_jax(x0, policy, T, X_basis, alpha, lengthscales, max_force=20.0):
    def step(x, _):
        a = jnp.dot(policy, x)
        a = jnp.clip(a, -max_force, max_force)  # clip to dataset range!
        features = build_features(x, a)
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

def plot_enhanced_time_series(traj, name, out_dir):
    """Enhanced version of plot_time_series with improved visuals."""
    dirs = _make_output_dirs(out_dir)
    t = np.arange(len(traj))
    labels = ["x", "x_dot", "theta", "theta_dot"]
    
    # Extract initial state for title
    initial_state = traj[0]
    initial_state_str = f"Initial: x={initial_state[0]:.2f}, θ={initial_state[2]:.2f}"
    
    # Create a descriptive title
    title = name.replace("_", " ").title()
    
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    for i in range(4):
        axs[i].plot(t, traj[:, i], linewidth=2.0)
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
        
        # Add horizontal line at y=0 for reference
        axs[i].axhline(y=0, color='r', linestyle='--', alpha=0.3)

    axs[-1].set_xlabel("Time step")
    fig.suptitle(f"{title}\n{initial_state_str}")
    plt.subplots_adjust(hspace=0.3)

    fig.savefig(os.path.join(dirs["pdf"], f"{name}.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(dirs["png"], f"{name}.png"), bbox_inches='tight')
    plt.close(fig)

def plot_superimposed_time_series(traj_model, traj_true, labels, fname=None, title="", policy=None):
    """Improved version with better styling and policy visualization."""
    T = traj_model.shape[0]
    steps = np.arange(T)
    
    # Calculate actions for both trajectories if policy is provided
    if policy is not None:
        actions_model = np.array([np.dot(policy, state) for state in traj_model])
        actions_true = np.array([np.dot(policy, state) for state in traj_true])
        # Create a figure with 5 subplots (4 states + action)
        fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    else:
        # Just 4 state variables
        fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Plot state variables
    for i, label in enumerate(labels):
        axs[i].plot(steps, traj_true[:, i], label="True System", color="tab:blue", lw=2.5)
        axs[i].plot(steps, traj_model[:, i], "--", label="Noisy Model", color="tab:orange", lw=2.5)
        axs[i].set_ylabel(label)
        axs[i].legend(loc='best')
        axs[i].grid(True)
        # Add zero reference line
        axs[i].axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    # Plot actions if available
    if policy is not None:
        axs[4].plot(steps, actions_true, label="True System", color="tab:blue", lw=2.5)
        axs[4].plot(steps, actions_model, "--", label="Noisy Model", color="tab:orange", lw=2.5)
        axs[4].set_ylabel("Action")
        axs[4].legend(loc='best')
        axs[4].grid(True)
        axs[4].axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    # Add policy information to title if available
    if policy is not None:
        policy_str = f"Policy: [{policy[0]:.2f}, {policy[1]:.2f}, {policy[2]:.2f}, {policy[3]:.2f}]"
        title = f"{title}\n{policy_str}"
    
    # Set title and labels
    fig.suptitle(title, fontsize=16)
    axs[-1].set_xlabel("Time step")
    plt.subplots_adjust(hspace=0.3, top=0.92)
    
    # Save figure if filename is provided
    if fname:
        plt.savefig(fname, bbox_inches='tight')
        plt.savefig(f"{os.path.splitext(fname)[0]}.pdf", bbox_inches='tight')
    
    plt.close(fig)

def optimise_and_plot(x0, sigma_l, X_basis, alpha, lengthscales, T, max_force, out_dir, tag):
    """Run optimization for a specific initial state and sigma_l combination."""
    p0 = np.zeros(4)
    
    def objective(p):
        traj = model_rollout_jax(x0, p, T, X_basis, alpha, lengthscales, max_force=max_force)
        return trajectory_loss_jax(traj, X0=jnp.zeros(4), sigma_l=sigma_l)

    obj_and_grad = jax.jit(jax.value_and_grad(objective))

    res = minimize(
        lambda p: float(obj_and_grad(p)[0]),
        p0,
        jac=lambda p: np.array(obj_and_grad(p)[1]),
        method="L-BFGS-B",
        options={"maxiter": 100}
    )
    
    print(f"=== {tag.upper()} ===")
    print("Initial state:", np.array(x0))
    print("Sigma_l:", np.array(sigma_l))
    print("Optimized policy (on noisy model):", res.x)
    print("Final noisy model rollout loss:", res.fun)

    # Plot model rollout (predicted) with enhanced plotting
    traj_model = np.array(model_rollout_jax(x0, res.x, T, X_basis, alpha, lengthscales, max_force=max_force))
    plot_enhanced_time_series(traj_model, f"NoisyModel_{tag}", out_dir)

    # Compare with true simulator (clip force to ±20!)
    from cartpole.simulation import rollout
    def linear_policy_fn(t, state):
        force = np.dot(res.x, state)
        return np.clip(force, -max_force, max_force)
    traj_true = rollout(np.array(x0), T=T, action_fn=linear_policy_fn, remap=True)
    plot_enhanced_time_series(traj_true, f"True_{tag}", out_dir)
    
    # Superimposed plot with enhanced styling and policy info
    labels = ["x", "x_dot", "theta", "theta_dot"]
    plot_superimposed_time_series(
        traj_model, traj_true, labels,
        fname=f"{out_dir}/comparison_{tag}.png",
        title=f"Noisy Model vs True System - {tag.replace('_', ' ').title()}",
        policy=res.x
    )
    
    # Calculate and plot policy actions over time
    actions_model = np.array([np.dot(res.x, state) for state in traj_model])
    actions_true = np.array([np.dot(res.x, state) for state in traj_true])
    
    # Plot actions
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(T), actions_true, label="True System", lw=2.5)
    plt.plot(np.arange(T), actions_model, '--', label="Noisy Model", lw=2.5)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.xlabel("Time step")
    plt.ylabel("Action")
    plt.title(f"Actions for {tag.replace('_', ' ').title()}: [{res.x[0]:.2f}, {res.x[1]:.2f}, {res.x[2]:.2f}, {res.x[3]:.2f}]")
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/policy_actions_{tag}.png", bbox_inches='tight')
    plt.close()
    
    return res, traj_model, traj_true

def main():
    out_dir = "figures/robustness/observation_noise/model_policy_jax"
    os.makedirs(out_dir, exist_ok=True)
    
    T = 20  # Longer horizon to see effects of noise
    max_force = 20.0

    # Load the kernel model fitted to noisy observations
    model = np.load("models/robustness/observation_noise/nonlinear_models.npz")
    X_basis = jnp.array(model["clean_X_basis"])
    alpha = jnp.array(model["noisy_alpha"])    # Use the noisy alpha for the noisy model
    lengthscales = jnp.array(model["clean_lengthscales"])
    
    print("Loaded kernel model fitted to noisy observations")
    
    # Define multiple sigma_l values (importance weights for different state dimensions)
    sigma_l_list = [
        jnp.array([0.5, 0.4, 0.05, 0.25]),  # Original from optimise_policy
        jnp.array([0.4, 0.3, 0.05, 0.2]),   # Alternative from optimise_policy
        jnp.array([0.3, 0.3, 0.03, 0.2]),   # More emphasis on angle stabilization
    ]
    
    # Define multiple initial conditions
    x0_list = [
        ("upright", jnp.array([0.0, 0.0, 0.1, 0.0])),
        ("tilted", jnp.array([0.0, 0.0, 0.3, 0.0])),
        ("velocity", jnp.array([0.0, 0.2, 0.1, 0.1])),
    ]
    
    # Store results for comparative visualization later
    all_results = {}
    
    # Run optimization for each combination
    for sigma_idx, sigma_l in enumerate(sigma_l_list):
        for x0_tag, x0 in x0_list:
            tag = f"{x0_tag}_sig{sigma_idx}"
            res, traj_model, traj_true = optimise_and_plot(
                x0, sigma_l, X_basis, alpha, lengthscales, T, max_force, out_dir, tag
            )
            all_results[tag] = (res.x, res.fun, traj_model, traj_true)
    
    # Save the best policy for further use
    best_policies = {}
    for tag, (policy, loss, _, _) in all_results.items():
        best_policies[tag] = {"policy": policy, "loss": loss}
    
    # Save all policies to a file
    np.savez(
        "models/robustness/observation_noise/optimized_policies.npz",
        **{k: v["policy"] for k, v in best_policies.items()}
    )
    
    # Generate a table of results
    result_text = "# Policy Optimisation Results (Kernel Model, Noisy Observations)\n\n"
    result_text += "| Scenario | Sigma_L | Policy | Loss |\n"
    result_text += "|----------|---------|--------|------|\n"
    
    for tag, (policy, loss, _, _) in all_results.items():
        parts = tag.split("_sig")
        scenario = parts[0]
        sig_idx = int(parts[1])
        sigma = sigma_l_list[sig_idx]
        policy_str = ", ".join([f"{p:.2f}" for p in policy])
        result_text += f"| {scenario} | {np.array2string(sigma, precision=2)} | [{policy_str}] | {loss:.4f} |\n"
    
    # Save the results table
    with open(f"{out_dir}/optimization_results.md", "w") as f:
        f.write(result_text)
    
    # Compare clean vs noisy model
    print("\n=== COMPARISON WITH CLEAN MODEL ===")
    # Load the clean action-conditioned model for comparison
    clean_model = np.load("models/policy_search/model_kernel_optimized_sincos_action.npz")
    clean_X_basis = jnp.array(clean_model["X_basis"])
    clean_alpha = jnp.array(clean_model["alpha"])
    clean_lengthscales = jnp.array(clean_model["lengthscales"])
    
    # For a selected scenario, compare performance of policies optimized on noisy vs clean
    selected_tag = "upright_sig0"  # Choose one scenario to analyze deeply
    selected_policy = all_results[selected_tag][0]
    selected_x0 = x0_list[0][1]  # Upright initial state
    
    # Rollout with noisy model
    noisy_traj = np.array(model_rollout_jax(
        selected_x0, selected_policy, T, X_basis, alpha, lengthscales, max_force=max_force
    ))
    
    # Rollout with clean model
    clean_traj = np.array(model_rollout_jax(
        selected_x0, selected_policy, T, clean_X_basis, clean_alpha, clean_lengthscales, max_force=max_force
    ))
    
    # Plot comparison between clean and noisy model predictions
    labels = ["x", "x_dot", "theta", "theta_dot"]
    
    # Calculate actions for both trajectories
    actions_clean = np.array([np.dot(selected_policy, state) for state in clean_traj])
    actions_noisy = np.array([np.dot(selected_policy, state) for state in noisy_traj])
    
    # Create figure with 5 subplots (4 states + action)
    fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    steps = np.arange(T)
    
    for i, label in enumerate(labels):
        axs[i].plot(steps, clean_traj[:, i], label="Clean Model", color="tab:green", lw=2.5)
        axs[i].plot(steps, noisy_traj[:, i], "--", label="Noisy Model", color="tab:purple", lw=2.5)
        axs[i].set_ylabel(label)
        axs[i].legend(loc='best')
        axs[i].grid(True)
        axs[i].axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    # Add action plot
    axs[4].plot(steps, actions_clean, label="Clean Model", color="tab:green", lw=2.5)
    axs[4].plot(steps, actions_noisy, "--", label="Noisy Model", color="tab:purple", lw=2.5)
    axs[4].set_ylabel("Action")
    axs[4].legend(loc='best')
    axs[4].grid(True)
    axs[4].axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    policy_str = f"Policy: [{selected_policy[0]:.2f}, {selected_policy[1]:.2f}, {selected_policy[2]:.2f}, {selected_policy[3]:.2f}]"
    fig.suptitle(f"Clean vs Noisy Model Predictions\n{policy_str}", fontsize=16)
    axs[-1].set_xlabel("Time step")
    plt.subplots_adjust(hspace=0.3, top=0.92)
    
    plt.savefig(f"{out_dir}/clean_vs_noisy_model.png", bbox_inches='tight')
    plt.savefig(f"{out_dir}/clean_vs_noisy_model.pdf", bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nAll optimizations complete. Results saved to {out_dir}")
    print(f"Best policies saved to models/robustness/observation_noise/optimized_policies.npz")

if __name__ == "__main__":
    main()
