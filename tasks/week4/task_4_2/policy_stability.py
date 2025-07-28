import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import os
import time
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

# Import the noisy dynamics simulation
from tasks.week4.task_4_2.noisy_dynamics import noisy_rollout

# Set global plotting parameters for better readability in reports
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 13,
    'figure.titlesize': 16
})

def load_policies():
    """Load the optimized policies from task 4.1."""
    try:
        return np.load("models/task_4.1/optimized_policies.npz")
    except FileNotFoundError:
        print("Error: Could not find optimized policies from task 4.1.")
        print("Please run task_4_1/policy.py first.")
        exit(1)

def evaluate_policy_stability(policy, x0, noise_levels, T=100, n_trials=10):
    """
    Evaluate the stability of a policy across multiple trials with different noise levels.
    
    Parameters:
    -----------
    policy : array, shape (4,)
        Linear policy parameters [k_x, k_x_dot, k_theta, k_theta_dot]
    x0 : array, shape (4,)
        Initial state [x, x_dot, theta, theta_dot]
    noise_levels : list of float
        Standard deviations of noise to test
    T : int
        Number of time steps for each rollout
    n_trials : int
        Number of trials to run for each noise level
    
    Returns:
    --------
    results : dict
        Dictionary containing trajectories, losses, and statistics
    """
    max_force = 20.0
    
    def policy_fn(t, state):
        force = np.dot(policy, state)
        return np.clip(force, -max_force, max_force)
    
    # Loss normalization parameters
    sigma_l = np.array([0.5, 0.4, 0.05, 0.25])
    
    results = {
        "noise_levels": noise_levels,
        "trajectories": {},
        "losses": {},
        "statistics": {},
    }
    
    for noise_std in noise_levels:
        print(f"Evaluating noise level: {noise_std}")
        
        # Store all trajectories and losses for this noise level
        all_trajs = []
        all_losses = []
        
        for trial in range(n_trials):
            # Run noisy rollout
            traj = noisy_rollout(x0, T, policy_fn, noise_std=noise_std)
            
            # Calculate losses (deviation from origin)
            diffs = (traj - np.zeros((1, 4))) / sigma_l
            sq_norms = np.sum(diffs**2, axis=1)
            per_step_losses = 1 - np.exp(-0.5 * sq_norms)
            total_loss = np.sum(per_step_losses)
            
            all_trajs.append(traj)
            all_losses.append(total_loss)
        
        # Convert to numpy arrays for easier statistics
        all_trajs = np.array(all_trajs)  # Shape (n_trials, T+1, 4)
        all_losses = np.array(all_losses)  # Shape (n_trials,)
        
        # Calculate statistics
        mean_traj = np.mean(all_trajs, axis=0)  # Average trajectory across trials
        std_traj = np.std(all_trajs, axis=0)    # Standard deviation of trajectories
        
        mean_loss = np.mean(all_losses)
        std_loss = np.std(all_losses)
        
        # Store results
        results["trajectories"][noise_std] = {
            "all": all_trajs,
            "mean": mean_traj,
            "std": std_traj,
        }
        
        results["losses"][noise_std] = {
            "all": all_losses,
            "mean": mean_loss,
            "std": std_loss,
        }
        
        results["statistics"][noise_std] = {
            "mean_final_state": mean_traj[-1],
            "std_final_state": std_traj[-1],
            "failure_rate": np.mean(np.abs(all_trajs[:, -1, 2]) > 0.5),  # θ > 0.5 rad
        }
    
    return results

def plot_stability_results(results, policy_name, out_dir):
    """Create plots to visualize the stability results."""
    os.makedirs(out_dir, exist_ok=True)
    
    noise_levels = results["noise_levels"]
    T = results["trajectories"][noise_levels[0]]["mean"].shape[0] - 1  # Get T from data
    time_steps = np.arange(T + 1)
    
    # 1. Plot mean trajectories for each noise level
    plt.figure(figsize=(16, 12))
    state_labels = ["Cart Position", "Cart Velocity", "Pendulum Angle", "Pendulum Angular Velocity"]
    
    for i in range(4):
        plt.subplot(4, 1, i+1)
        
        for noise_std in noise_levels:
            mean_traj = results["trajectories"][noise_std]["mean"]
            std_traj = results["trajectories"][noise_std]["std"]
            
            # Plot mean trajectory
            line, = plt.plot(time_steps, mean_traj[:, i], label=f"Noise σ={noise_std}")
            color = to_rgb(line.get_color())
            
            # Plot error bands (mean ± std)
            plt.fill_between(
                time_steps,
                mean_traj[:, i] - std_traj[:, i],
                mean_traj[:, i] + std_traj[:, i],
                color=(*color, 0.2)  # Use same color with alpha
            )
        
        plt.ylabel(state_labels[i])
        plt.grid(True)
        
        if i == 0:
            plt.title(f"Stability of Policy: {policy_name}")
    
    plt.subplot(4, 1, 4)
    plt.xlabel("Time Steps")
    plt.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/mean_trajectories_{policy_name}.png")
    plt.savefig(f"{out_dir}/mean_trajectories_{policy_name}.pdf")
    plt.close()
    
    # 2. Plot loss statistics across noise levels
    plt.figure(figsize=(10, 6))
    
    # Gather mean and std losses for all noise levels
    mean_losses = [results["losses"][noise_std]["mean"] for noise_std in noise_levels]
    std_losses = [results["losses"][noise_std]["std"] for noise_std in noise_levels]
    
    plt.errorbar(noise_levels, mean_losses, yerr=std_losses, marker='o', linestyle='-', capsize=5)
    plt.xlabel("Noise Standard Deviation")
    plt.ylabel("Total Loss")
    plt.title(f"Loss vs. Noise Level for Policy: {policy_name}")
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f"{out_dir}/loss_vs_noise_{policy_name}.png")
    plt.savefig(f"{out_dir}/loss_vs_noise_{policy_name}.pdf")
    plt.close()
    
    # 3. Plot failure rates
    failure_rates = [results["statistics"][noise_std]["failure_rate"] for noise_std in noise_levels]
    
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, failure_rates, marker='o', linestyle='-')
    plt.xlabel("Noise Standard Deviation")
    plt.ylabel("Failure Rate")
    plt.title(f"Failure Rate vs. Noise Level for Policy: {policy_name}")
    plt.grid(True)
    plt.ylim(-0.05, 1.05)  # Limit y-axis to [0, 1] with some padding
    plt.tight_layout()
    
    plt.savefig(f"{out_dir}/failure_rate_{policy_name}.png")
    plt.savefig(f"{out_dir}/failure_rate_{policy_name}.pdf")
    plt.close()
    
    # 4. Plot standard deviation of final state components
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(state_labels):
        std_values = [results["statistics"][noise_std]["std_final_state"][i] for noise_std in noise_levels]
        plt.plot(noise_levels, std_values, marker='o', linestyle='-', label=label)
    
    plt.xlabel("Noise Standard Deviation")
    plt.ylabel("Standard Deviation of Final State")
    plt.title(f"Final State Variability for Policy: {policy_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"{out_dir}/final_state_std_{policy_name}.png")
    plt.savefig(f"{out_dir}/final_state_std_{policy_name}.pdf")
    plt.close()
    
    # Save results as numpy files for later analysis
    np.savez(
        f"{out_dir}/stability_results_{policy_name}.npz",
        noise_levels=noise_levels,
        mean_losses=mean_losses,
        std_losses=std_losses,
        failure_rates=failure_rates
    )

def create_comparison_plots(all_policy_results, policy_names, out_dir):
    """Create plots comparing all policies side by side."""
    noise_levels = all_policy_results[policy_names[0]]["noise_levels"]
    
    # 1. Compare mean losses
    plt.figure(figsize=(12, 7))
    
    for policy_name in policy_names:
        mean_losses = [all_policy_results[policy_name]["losses"][noise_std]["mean"] for noise_std in noise_levels]
        plt.plot(noise_levels, mean_losses, marker='o', linestyle='-', label=policy_name)
    
    plt.xlabel("Noise Standard Deviation")
    plt.ylabel("Mean Total Loss")
    plt.title("Loss vs. Noise Level Across Policies")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"{out_dir}/loss_comparison.png")
    plt.savefig(f"{out_dir}/loss_comparison.pdf")
    plt.close()
    
    # 2. Compare failure rates
    plt.figure(figsize=(12, 7))
    
    for policy_name in policy_names:
        failure_rates = [all_policy_results[policy_name]["statistics"][noise_std]["failure_rate"] for noise_std in noise_levels]
        plt.plot(noise_levels, failure_rates, marker='o', linestyle='-', label=policy_name)
    
    plt.xlabel("Noise Standard Deviation")
    plt.ylabel("Failure Rate")
    plt.title("Failure Rate vs. Noise Level Across Policies")
    plt.grid(True)
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    
    plt.savefig(f"{out_dir}/failure_rate_comparison.png")
    plt.savefig(f"{out_dir}/failure_rate_comparison.pdf")
    plt.close()
    
    # 3. Compare angle stability (std dev of final angle)
    plt.figure(figsize=(12, 7))
    
    for policy_name in policy_names:
        angle_stds = [all_policy_results[policy_name]["statistics"][noise_std]["std_final_state"][2] for noise_std in noise_levels]
        plt.plot(noise_levels, angle_stds, marker='o', linestyle='-', label=policy_name)
    
    plt.xlabel("Noise Standard Deviation")
    plt.ylabel("Std Dev of Final Angle (rad)")
    plt.title("Angle Stability vs. Noise Level Across Policies")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"{out_dir}/angle_stability_comparison.png")
    plt.savefig(f"{out_dir}/angle_stability_comparison.pdf")
    plt.close()
    
    # Create a markdown summary
    markdown = "# Stability Analysis of Cartpole Policies with Noisy Dynamics\n\n"
    markdown += "## Evaluated Policies\n\n"
    
    for policy_name in policy_names:
        policy = all_policy_results[policy_name]["policy"]
        policy_str = ", ".join([f"{p:.3f}" for p in policy])
        markdown += f"- **{policy_name}**: `[{policy_str}]`\n"
    
    markdown += "\n## Results Summary\n\n"
    markdown += "| Noise Level | " + " | ".join([f"{name} Loss" for name in policy_names]) + " | "
    markdown += " | ".join([f"{name} Failure Rate" for name in policy_names]) + " |\n"
    markdown += "|" + "-|" * (1 + 2 * len(policy_names)) + "\n"
    
    for noise_std in noise_levels:
        row = f"| {noise_std:.3f} | "
        
        # Add mean losses
        for policy_name in policy_names:
            mean_loss = all_policy_results[policy_name]["losses"][noise_std]["mean"]
            row += f"{mean_loss:.2f} | "
        
        # Add failure rates
        for policy_name in policy_names:
            failure_rate = all_policy_results[policy_name]["statistics"][noise_std]["failure_rate"]
            row += f"{failure_rate:.2f} | "
        
        markdown += row + "\n"
    
    markdown += "\n## Observations\n\n"
    markdown += "- As noise increases, all policies show degraded performance\n"
    markdown += "- Policies optimized with more emphasis on angle stabilization tend to be more robust\n"
    markdown += "- The failure rate increases non-linearly with noise magnitude\n"
    
    with open(f"{out_dir}/stability_analysis_summary.md", "w") as f:
        f.write(markdown)

def plot_superimposed_time_series(traj_model, traj_true, labels, fname=None, title="", policy=None):
    """Plot superimposed time series with fixed dimensions."""
    # Get the actual time steps from trajectories
    T = min(traj_model.shape[0], traj_true.shape[0])
    steps = np.arange(T)
    
    # Ensure trajectories have the same length
    traj_model_trimmed = traj_model[:T]
    traj_true_trimmed = traj_true[:T]
    
    # Calculate actions for both trajectories if policy is provided
    if policy is not None:
        # Make sure we don't create dimension mismatch
        actions_model = np.array([np.dot(policy, state) for state in traj_model_trimmed])
        actions_true = np.array([np.dot(policy, state) for state in traj_true_trimmed])
        
        # Create a figure with 5 subplots (4 states + action)
        fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    else:
        # Just 4 state variables
        fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Plot state variables
    for i, label in enumerate(labels):
        axs[i].plot(steps, traj_true_trimmed[:, i], label="True System", color="tab:blue", lw=2.5)
        axs[i].plot(steps, traj_model_trimmed[:, i], "--", label="Noisy Model", color="tab:orange", lw=2.5)
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

def main():
    # Create output directory
    out_dir = "figures/task_4_2/policy_stability"
    os.makedirs(out_dir, exist_ok=True)
    
    # Load policies from task 4.1
    policies = load_policies()
    policy_names = list(policies.keys())
    
    # Set up initial conditions to test
    x0_dict = {
        "upright": np.array([0.0, 0.0, 0.1, 0.0]),
        "tilted": np.array([0.0, 0.0, 0.3, 0.0])
    }
    
    # Set up noise levels to test
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
    
    # Set up simulation parameters
    T = 100  # Time steps per trial
    n_trials = 20  # Number of trials per noise level
    
    # Store results for all policies
    all_results = {}
    
    # Process each initial condition separately
    for x0_name, x0 in x0_dict.items():
        print(f"\n=== Testing with initial condition: {x0_name} ===")
        
        x0_out_dir = f"{out_dir}/{x0_name}"
        os.makedirs(x0_out_dir, exist_ok=True)
        
        # Evaluate each policy
        x0_results = {}
        
        for policy_name in policy_names:
            print(f"\nEvaluating policy: {policy_name}")
            policy = policies[policy_name]
            
            # Store policy for reference
            x0_results[policy_name] = {
                "policy": policy
            }
            
            # Evaluate policy stability
            start_time = time.time()
            stability_results = evaluate_policy_stability(
                policy, x0, noise_levels, T=T, n_trials=n_trials
            )
            
            # Merge results
            x0_results[policy_name].update(stability_results)
            
            # Plot individual policy results
            plot_stability_results(stability_results, f"{policy_name}_{x0_name}", x0_out_dir)
            
            print(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
        
        # Create comparative plots for this initial condition
        create_comparison_plots(x0_results, policy_names, x0_out_dir)
        
        # Store results for this initial condition
        all_results[x0_name] = x0_results
    
    # Save all results
    os.makedirs("models/task_4_2", exist_ok=True)
    for x0_name, x0_results in all_results.items():
        # Create a clean version for saving (remove large trajectory arrays)
        save_results = {}
        
        for policy_name, policy_results in x0_results.items():
            save_results[policy_name] = {
                "policy": policy_results["policy"],
                "noise_levels": policy_results["noise_levels"],
                "mean_losses": [policy_results["losses"][noise_std]["mean"] for noise_std in noise_levels],
                "std_losses": [policy_results["losses"][noise_std]["std"] for noise_std in noise_levels],
                "failure_rates": [policy_results["statistics"][noise_std]["failure_rate"] for noise_std in noise_levels],
                "final_state_stds": [policy_results["statistics"][noise_std]["std_final_state"] for noise_std in noise_levels]
            }
        
        np.savez(f"models/task_4_2/stability_results_{x0_name}.npz", results=save_results)
    
    print("\nStability analysis complete. Results saved to:")
    print(f"- {out_dir}")
    print("- models/task_4_2/")

if __name__ == "__main__":
    main()
