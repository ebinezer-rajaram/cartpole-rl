import numpy as np
import os
from tasks.week4.task_4_2.noisy_dynamics import noisy_rollout
import matplotlib.pyplot as plt

def collect_dataset(n_trajectories=100, steps_per_trajectory=20, noise_std=0.0):
    """
    Collect a dataset of state-action-nextstate from noisy dynamics.
    
    Parameters:
    -----------
    n_trajectories : int
        Number of trajectories to collect
    steps_per_trajectory : int
        Number of steps per trajectory
    noise_std : float
        Standard deviation of noise in dynamics
        
    Returns:
    --------
    X : array, shape (n_samples, 5)
        State-action pairs [x, x_dot, theta, theta_dot, action]
    Y : array, shape (n_samples, 4)
        State changes [dx, dx_dot, dtheta, dtheta_dot]
    """
    n_samples = n_trajectories * steps_per_trajectory
    X = np.zeros((n_samples, 5))  # state-action pairs
    Y = np.zeros((n_samples, 4))  # state changes
    
    sample_idx = 0
    
    # Define possible initial states with some variability
    x_range = (-1, 1)
    x_dot_range = (-1, 1)
    theta_range = (-0.3, 0.3)  # small angles to avoid instability
    theta_dot_range = (-0.5, 0.5)
    
    # Random exploration policy
    def random_policy(t, state):
        return np.random.uniform(-20, 20)
    
    for traj in range(n_trajectories):
        # Random initial state
        x0 = np.array([
            np.random.uniform(*x_range),
            np.random.uniform(*x_dot_range),
            np.random.uniform(*theta_range),
            np.random.uniform(*theta_dot_range)
        ])
        
        # Run a trajectory with noisy dynamics
        traj_states = noisy_rollout(x0, steps_per_trajectory, random_policy, noise_std=noise_std)
        
        # Extract state-action-nextstate tuples
        for t in range(steps_per_trajectory):
            state = traj_states[t]
            next_state = traj_states[t+1]
            
            # Get the action that was applied (we need to recompute it)
            action = random_policy(t, state)
            
            # Store state-action pair
            X[sample_idx, :4] = state
            X[sample_idx, 4] = action
            
            # Store state change
            Y[sample_idx] = next_state - state
            
            sample_idx += 1
    
    return X, Y

def main():
    """Generate datasets with noisy dynamics for Task 4.2."""
    np.random.seed(42)
    
    # Create output directory
    os.makedirs("data/task_4.2", exist_ok=True)
    
    # Generate dataset with clean dynamics (noise_std=0)
    print("Generating dataset with clean dynamics...")
    X_clean, Y_clean = collect_dataset(n_trajectories=100, steps_per_trajectory=20, noise_std=0.0)
    
    # Generate datasets with different noise levels
    noise_levels = [0.01, 0.05, 0.1]
    
    for noise_std in noise_levels:
        print(f"Generating dataset with noise_std={noise_std}...")
        X_noisy, Y_noisy = collect_dataset(n_trajectories=100, steps_per_trajectory=20, noise_std=noise_std)
        
        # Save dataset
        np.savez(
            f"data/task_4.2/nonlinear_dataset_noise_{noise_std:.2f}.npz",
            X=X_noisy, Y=Y_noisy, noise_std=noise_std
        )
        
        print(f"  - Dataset saved with {X_noisy.shape[0]} samples")
        
        # Create a simple visualization to verify the dataset
        plt.figure(figsize=(12, 8))
        
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.scatter(X_clean[:, i], Y_clean[:, i], alpha=0.3, label='Clean')
            plt.scatter(X_noisy[:, i], Y_noisy[:, i], alpha=0.3, label='Noisy')
            labels = ["x", "x_dot", "theta", "theta_dot"]
            plt.xlabel(labels[i])
            plt.ylabel(f"d{labels[i]}")
            plt.title(f"{labels[i]} vs d{labels[i]}, noise={noise_std}")
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        os.makedirs("figures/task_4.2/data", exist_ok=True)
        plt.savefig(f"figures/task_4.2/data/dataset_comparison_noise_{noise_std:.2f}.png")
        plt.close()
    
    # Save the clean dataset for reference
    np.savez("data/task_4.2/nonlinear_dataset_clean.npz", X=X_clean, Y=Y_clean, noise_std=0.0)
    print(f"Clean dataset saved with {X_clean.shape[0]} samples")
    
    # Create a comparison dataset for task 4.2
    main_noise_std = 0.05  # Use this as the main dataset for task 4.2
    np.savez("models/task_4.2/nonlinear_dataset.npz", X=X_noisy, Y=Y_noisy, noise_std=main_noise_std)
    print(f"Main dataset for task 4.2 saved with noise_std={main_noise_std}")
    
    print("\nData generation complete for Task 4.2!")

if __name__ == "__main__":
    main()
