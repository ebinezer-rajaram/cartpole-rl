import numpy as np
from cartpole.simulation import rollout as clean_rollout
import os
import matplotlib.pyplot as plt

# Define our own step function since it can't be imported directly
def clean_step(state, action, dt=0.1, clip=False):
    """
    Implement the cartpole dynamics step function.
    
    Parameters:
    -----------
    state : array-like, shape (4,)
        The current state [x, x_dot, theta, theta_dot]
    action : float
        The force to apply
    dt : float
        Time step size
    clip : bool
        Whether to clip the action to [-20, 20]
        
    Returns:
    --------
    next_state : array, shape (4,)
        The next state after applying action
    """
    # Constants from the cartpole system
    gravity = 9.81
    cart_mass = 1.0
    pole_mass = 0.1
    pole_length = 0.5
    friction = 0.1
    
    # Clip action if requested
    if clip:
        action = np.clip(action, -20.0, 20.0)
    
    # Extract state variables
    x, x_dot, theta, theta_dot = state
    
    # Calculate the accelerations using cartpole dynamics equations
    total_mass = cart_mass + pole_mass
    pole_mass_length = pole_mass * pole_length
    
    # Force from the pole on the cart
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    temp = (action + pole_mass_length * theta_dot**2 * sin_theta - friction * x_dot) / total_mass
    theta_acc = (gravity * sin_theta - cos_theta * temp) / (pole_length * (4.0/3.0 - pole_mass * cos_theta**2 / total_mass))
    x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass
    
    # Update state using Euler integration
    x_new = x + dt * x_dot
    x_dot_new = x_dot + dt * x_acc
    theta_new = theta + dt * theta_dot
    theta_dot_new = theta_dot + dt * theta_acc
    
    # Remap theta to [-pi, pi]
    theta_new = ((theta_new + np.pi) % (2 * np.pi)) - np.pi
    
    return np.array([x_new, x_dot_new, theta_new, theta_dot_new])

def noisy_step(state, action, noise_std=0.0, dt=0.1, clip=False):
    """
    Add noise to the cartpole dynamics step function.
    
    Parameters:
    -----------
    state : array-like, shape (4,)
        The current state [x, x_dot, theta, theta_dot]
    action : float
        The force to apply
    noise_std : float
        Standard deviation of the Gaussian noise to add to state derivatives
    dt : float
        Time step size
    clip : bool
        Whether to clip the action to [-20, 20]
        
    Returns:
    --------
    next_state : array, shape (4,)
        The next state after applying action and adding noise
    """
    # Call the clean step function
    next_state = clean_step(state, action, dt=dt, clip=clip)
    
    # Add Gaussian noise to the state derivatives
    # We apply noise to the derivatives (rates of change), not directly to position
    delta_state = next_state - state
    
    # Add noise proportional to the state derivatives
    noise = np.random.normal(0, noise_std, size=delta_state.shape)
    noisy_delta = delta_state + noise
    
    # Calculate the next state with noise
    noisy_next_state = state + noisy_delta
    
    # Ensure the angle stays within [-π, π]
    noisy_next_state[2] = ((noisy_next_state[2] + np.pi) % (2 * np.pi)) - np.pi
    
    return noisy_next_state

def noisy_rollout(init_state, T, action_fn, noise_std=0.0, dt=0.1, remap=True):
    """
    Simulate the cartpole with noisy dynamics for T steps.
    
    Parameters:
    -----------
    init_state : array-like, shape (4,)
        The initial state [x, x_dot, theta, theta_dot]
    T : int
        Number of time steps to simulate
    action_fn : callable
        Function that takes (t, state) and returns an action
    noise_std : float
        Standard deviation of the noise to add at each step
    dt : float
        Time step size
    remap : bool
        Whether to remap the angle to [-π, π] at each step
        
    Returns:
    --------
    trajectory : array, shape (T+1, 4)
        The state trajectory, including the initial state
    """
    state = np.array(init_state, dtype=np.float64)
    trajectory = np.zeros((T + 1, 4))
    trajectory[0] = state
    
    for t in range(T):
        action = action_fn(t, state)
        state = noisy_step(state, action, noise_std=noise_std, dt=dt, clip=True)
        
        if remap:
            # Remap angle to [-π, π]
            state[2] = ((state[2] + np.pi) % (2 * np.pi)) - np.pi
        
        trajectory[t + 1] = state
    
    return trajectory

def test_noisy_dynamics():
    """Test the noisy dynamics implementation."""
    
    # Create output directories
    os.makedirs("figures/robustness/process_noise/noisy_dynamics", exist_ok=True)
    
    # Initial state (upright pendulum with small perturbation)
    init_state = np.array([0.0, 0.0, 0.1, 0.0])
    T = 50  # Number of time steps
    
    # Simple controller that applies constant force
    def constant_action(t, state):
        return 5.0
    
    # Compare clean vs. noisy dynamics with different noise levels
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    
    plt.figure(figsize=(16, 12))
    
    for i, noise_std in enumerate(noise_levels):
        if noise_std == 0.0:
            # Use clean rollout for the baseline
            traj = noisy_rollout(init_state, T, constant_action, noise_std=0.0)
            label = "Clean Dynamics"
            linestyle = "-"
            alpha = 1.0
        else:
            # Use noisy rollout
            traj = noisy_rollout(init_state, T, constant_action, noise_std=noise_std)
            label = f"Noise σ={noise_std}"
            linestyle = "--"
            alpha = 0.9 - i * 0.15
        
        # Plot the state variables
        state_labels = ["Cart Position", "Cart Velocity", "Pendulum Angle", "Pendulum Angular Velocity"]
        
        for j in range(4):
            plt.subplot(4, 1, j+1)
            plt.plot(np.arange(T+1), traj[:, j], label=label, linestyle=linestyle, alpha=alpha)
            plt.ylabel(state_labels[j])
            plt.grid(True)
            if j == 0:
                plt.title("Cartpole Dynamics with Different Noise Levels")
    
    # Add legends and adjust layout
    for j in range(4):
        plt.subplot(4, 1, j+1)
        plt.legend(loc="upper right")
    
    plt.subplot(4, 1, 4)
    plt.xlabel("Time Steps")
    
    plt.tight_layout()
    plt.savefig("figures/robustness/process_noise/noisy_dynamics/noise_comparison.png")
    plt.savefig("figures/robustness/process_noise/noisy_dynamics/noise_comparison.pdf")
    plt.close()
    
    print("Noise comparison test complete. Figures saved to figures/robustness/process_noise/noisy_dynamics/")

if __name__ == "__main__":
    test_noisy_dynamics()
