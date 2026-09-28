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
        # No need to remap angle when using sin/cos features
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

def rollout_process_noise(init_state, T=20, action_fn=None, remap=False, visual=False, noise_std=0.05):
    from cartpole.CartPole import CartPole, remap_angle
    env = CartPole(visual=visual)
    env.setState(init_state)
    traj = []
    for t in range(T):
        state = env.getState()
        action = 0.0 if action_fn is None else action_fn(t, state)
        env.performAction(action)
        next_state = env.getState()
        # Don't remap angle when using sin/cos features
        # Add process noise at each step
        next_state = next_state + np.random.normal(0, noise_std, size=next_state.shape)
        env.setState(next_state)
        traj.append(next_state.copy())
    return np.array(traj)


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
        axs[i].plot(steps, traj_model[:, i], "--", label="Model Prediction", color="tab:orange", lw=2.5)
        axs[i].set_ylabel(label)
        axs[i].legend(loc='best')
        axs[i].grid(True)
        # Add zero reference line
        axs[i].axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    # Plot actions if available
    if policy is not None:
        axs[4].plot(steps, actions_true, label="True System", color="tab:blue", lw=2.5)
        axs[4].plot(steps, actions_model, "--", label="Model Prediction", color="tab:orange", lw=2.5)
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
    print("Optimized policy:", res.x)
    print("Final model rollout loss:", res.fun)

    # Plot model rollout with enhanced plotting
    traj_model = np.array(model_rollout_jax(x0, res.x, T, X_basis, alpha, lengthscales, max_force=max_force))
    plot_enhanced_time_series(traj_model, f"Model_{tag}", out_dir)

    # Compare with true simulator (clip force to ±20!)
    from cartpole.simulation import rollout
    def linear_policy_fn(t, state):
        force = np.dot(res.x, state)
        return np.clip(force, -max_force, max_force)
    traj_true = rollout(np.array(x0), T=T, action_fn=linear_policy_fn, remap=False)
    plot_enhanced_time_series(traj_true, f"True_{tag}", out_dir)
    
    # Superimposed plot with enhanced styling and policy info
    labels = ["x", "x_dot", "theta", "theta_dot"]
    plot_superimposed_time_series(
        traj_model, traj_true, labels,
        fname=f"{out_dir}/comparison_{tag}.png",
        title=f"Model vs True System - {tag.replace('_', ' ').title()}",
        policy=res.x
    )
    
    # Calculate and plot policy actions over time
    actions_model = np.array([np.dot(res.x, state) for state in traj_model])
    actions_true = np.array([np.dot(res.x, state) for state in traj_true])
    
    # Plot actions
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(T), actions_true, label="True System", lw=2.5)
    plt.plot(np.arange(T), actions_model, '--', label="Model Prediction", lw=2.5)
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

def plot_multi_trajectory_comparison(trajectories, labels, noise_levels, title, out_dir, filename_base):
    """Plot multiple trajectories for comparison with different noise levels."""
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    state_labels = ["x", "x_dot", "theta", "theta_dot"]
    T = trajectories[0].shape[0]
    steps = np.arange(T)
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    
    for i, label in enumerate(state_labels):
        for j, (traj, traj_label, noise_std) in enumerate(zip(trajectories, labels, noise_levels)):
            linestyle = '-' if j == 0 else '--'
            axs[i].plot(steps, traj[:, i], linestyle=linestyle, color=colors[j], 
                        label=f"{traj_label} (noise={noise_std})", linewidth=2.0)
        
        axs[i].set_ylabel(label)
        axs[i].grid(True)
        if i == 0:
            axs[i].legend(loc='best')
        axs[i].axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    axs[-1].set_xlabel("Time step")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save figures
    dirs = _make_output_dirs(out_dir)
    plt.savefig(f"{dirs['pdf']}/{filename_base}.pdf", bbox_inches='tight')
    plt.savefig(f"{dirs['png']}/{filename_base}.png", bbox_inches='tight')
    plt.close(fig)

def main():
    out_dir = "figures/robustness/process_noise/model_policy"
    os.makedirs(out_dir, exist_ok=True)
    
    # Match the noise level used in data collection (from data.py)
    process_noise_std = 0.001
    T = 30
    max_force = 20.0

    model_path = "models/policy_search/nonlinear_model.npz"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run nonlinear.py first to generate the model.")
        return

    model = np.load(model_path)
    X_basis = jnp.array(model["X_basis"])
    alpha = jnp.array(model["alpha"])
    lengthscales = jnp.array(model["lengthscales"])
    
    print("Loaded kernel model trained on process-noise data")
    print(f"Model has {len(X_basis)} basis points")
    print(f"Lengthscales: {lengthscales}")
    
    sigma_l_list = [
        jnp.array([0.5, 0.4, 0.05, 0.25]),  # Balanced
        # jnp.array([0.3, 0.3, 0.03, 0.2]),   # More emphasis on angle stabilization
        # jnp.array([0.1, 0.1, 0.01, 0.1]),   # Very strict stabilization
    ]

    x0_list = [
        ("upright", jnp.array([0.0, 0.0, 0.1, 0.0])),
        # ("tilted", jnp.array([0.0, 0.0, 0.3, 0.0])),
        # ("displaced", jnp.array([1.0, 0.0, 0.1, 0.0])),
        # ("velocity", jnp.array([0.0, 0.5, 0.1, 0.1])),
    ]
    
    all_results = {}
    
    for sigma_idx, sigma_l in enumerate(sigma_l_list):
        for x0_tag, x0 in x0_list:
            tag = f"{x0_tag}_sig{sigma_idx}"
            res, traj_model, traj_true = optimise_and_plot(
                x0, sigma_l, X_basis, alpha, lengthscales, T, max_force, out_dir, tag
            )
            all_results[tag] = (res.x, res.fun, traj_model, traj_true)
    
    best_policies = {}
    for tag, (policy, loss, _, _) in all_results.items():
        best_policies[tag] = {"policy": policy, "loss": loss}
    
    np.savez(
        "models/robustness/process_noise/optimized_policies.npz",
        **{k: v["policy"] for k, v in best_policies.items()}
    )
    
    result_text = "# Policy Optimisation Results (Process Noise Model)\n\n"
    result_text += "| Scenario | Sigma_L | Policy | Loss |\n"
    result_text += "|----------|---------|--------|------|\n"
    
    for tag, (policy, loss, _, _) in all_results.items():
        parts = tag.split("_sig")
        scenario = parts[0]
        sig_idx = int(parts[1])
        sigma = sigma_l_list[sig_idx]
        policy_str = ", ".join([f"{p:.2f}" for p in policy])
        result_text += f"| {scenario} | {np.array2string(sigma, precision=2)} | [{policy_str}] | {loss:.4f} |\n"
    
    with open(f"{out_dir}/optimization_results.md", "w") as f:
        f.write(result_text)
    
    reference_policy = all_results["upright_sig0"][0]
    
    plt.figure(figsize=(12, 8))
    
    for i, (x0_tag, x0) in enumerate(x0_list):
        
        def linear_policy_fn(t, state):
            force = np.dot(reference_policy, state)  # Fixed: was using res.x which isn't defined here
            return np.clip(force, -max_force, max_force)

        traj_true_noisy = rollout_process_noise(np.array(x0), T=T, action_fn=linear_policy_fn, remap=False, noise_std=0.001)
        plot_enhanced_time_series(traj_true_noisy, f"TrueNoisy_{x0_tag}", out_dir)  # Fixed: use x0_tag instead of undefined tag
    
    plt.suptitle(f"Reference Policy Applied to Different Initial Conditions\nPolicy: [{reference_policy[0]:.2f}, {reference_policy[1]:.2f}, {reference_policy[2]:.2f}, {reference_policy[3]:.2f}]")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/policy_comparison_init_conditions.png", bbox_inches='tight')
    plt.savefig(f"{out_dir}/policy_comparison_init_conditions.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"\nAll optimizations complete. Results saved to {out_dir}")
    print(f"Best policies saved to models/robustness/process_noise/optimized_policies.npz")

    # After optimizing policies, create additional noise level comparisons
    reference_tag = "upright_sig0"
    reference_policy = all_results[reference_tag][0]
    reference_x0 = x0_list[0][1]  # upright initial state
    
    print("\nGenerating noise level comparison plots...")
    
    # Run model predictions (deterministic)
    model_traj = np.array(model_rollout_jax(reference_x0, reference_policy, T, X_basis, alpha, lengthscales, max_force=max_force))
    
    # Run true system with different noise levels
    def linear_policy_fn(t, state):
        force = np.dot(reference_policy, state)
        return np.clip(force, -max_force, max_force)
    
    # Create multiple rollouts with different noise levels
    noise_levels = [0.0, 0.001, 0.01, 0.05]
    true_trajs = []
    traj_labels = []
    
    # First trajectory is completely deterministic (from simulation.rollout)
    from cartpole.simulation import rollout
    true_clean = rollout(np.array(reference_x0), T=T, action_fn=linear_policy_fn, remap=False)
    true_trajs.append(true_clean)
    traj_labels.append("True (no noise)")
    
    # Add noisy trajectories with increasing noise levels
    for noise_std in noise_levels[1:]:
        true_noisy = rollout_process_noise(np.array(reference_x0), T=T, action_fn=linear_policy_fn, remap=False, noise_std=noise_std)
        true_trajs.append(true_noisy)
        traj_labels.append(f"True (with noise)")
    
    # Create detailed comparison plot
    trajectories = [model_traj] + true_trajs
    all_labels = ["Model Prediction"] + traj_labels
    all_noise_levels = ["N/A"] + [str(n) for n in noise_levels]
    
    plot_multi_trajectory_comparison(
        trajectories, all_labels, all_noise_levels,
        f"Model vs True System with Different Noise Levels\nPolicy: [{reference_policy[0]:.2f}, {reference_policy[1]:.2f}, {reference_policy[2]:.2f}, {reference_policy[3]:.2f}]",
        out_dir, "noise_level_comparison"
    )
    
    # Create an additional plot focusing only on the appropriate noise level (0.01)
    focus_traj = rollout_process_noise(np.array(reference_x0), T=T, action_fn=linear_policy_fn, remap=False, noise_std=process_noise_std)
    plot_superimposed_time_series(
        model_traj, focus_traj, ["x", "x_dot", "theta", "theta_dot"],
        fname=f"{out_dir}/model_vs_true_noise_{process_noise_std}.png", 
        title=f"Model vs True System with Process Noise (std={process_noise_std})",
        policy=reference_policy
    )
    
    # Run multiple trials with the same noise level to show variability
    print("Generating multiple trial comparisons...")
    trial_trajs = []
    for i in range(5):
        trial = rollout_process_noise(np.array(reference_x0), T=T, action_fn=linear_policy_fn, remap=False, noise_std=process_noise_std)
        trial_trajs.append(trial)
    
    # Plot multiple trials to show stochasticity
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    state_labels = ["x", "x_dot", "theta", "theta_dot"]
    steps = np.arange(T)
    
    for i, label in enumerate(state_labels):
        # Plot model prediction
        axs[i].plot(steps, model_traj[:, i], 'k-', label="Model Prediction", linewidth=2.5)
        
        # Plot multiple trials with the same noise level
        for j, trial in enumerate(trial_trajs):
            axs[i].plot(steps, trial[:, i], '--', alpha=0.5, linewidth=1.5, 
                      label=f"True Trial {j+1}" if i == 0 else None)
        
        axs[i].set_ylabel(label)
        axs[i].grid(True)
        if i == 0:
            axs[i].legend(loc='best')
        axs[i].axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    axs[-1].set_xlabel("Time step")
    fig.suptitle(f"Model vs Multiple True System Trials (noise={process_noise_std})\n" + 
                f"Policy: [{reference_policy[0]:.2f}, {reference_policy[1]:.2f}, {reference_policy[2]:.2f}, {reference_policy[3]:.2f}]", 
                fontsize=16)
    plt.tight_layout()
    
    dirs = _make_output_dirs(out_dir)
    plt.savefig(f"{dirs['pdf']}/multiple_trial_comparison.pdf", bbox_inches='tight')
    plt.savefig(f"{dirs['png']}/multiple_trial_comparison.png", bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nAll visualizations complete. Additional comparison plots saved to {out_dir}")

if __name__ == "__main__":
    main()
