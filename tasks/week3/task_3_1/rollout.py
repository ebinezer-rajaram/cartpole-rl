import numpy as np
import matplotlib.pyplot as plt
import os

from cartpole.simulation import rollout
from cartpole.CartPole import remap_angle
from .regression import predict_kernel
from cartpole.plotting import _make_output_dirs

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

def model_predict_fn(state, action, X_basis, alpha, lengthscales):
    x, x_dot, theta, theta_dot = state
    features = np.array([
        x,
        x_dot,
        np.sin(theta),
        np.cos(theta),
        theta_dot,
        action
    ])[None, :]
    return predict_kernel(features, X_basis, alpha, lengthscales)[0]

def model_rollout(x0, alpha, basis_X, lengthscales, T, action_fn):
    X = np.zeros((T, 4))
    x = x0.copy()
    for t in range(T):
        X[t] = x
        a = action_fn(t, x)
        dx = model_predict_fn(x, a, basis_X, alpha, lengthscales)
        x = np.array(x + dx)  # ensure NumPy array for mutation
        x[2] = remap_angle(x[2])
    return X

def main():
    labels = ["x", "x_dot", "theta", "theta_dot"]
    x0 = np.array([0.0, 0.0, 0.1, 0.0])
    T = 50
    
    # Define action functions with different amplitudes
    def create_action_fn(amplitude):
        return lambda t, state: np.sin(t / 5.0) * amplitude
    
    # Create action functions with amplitudes 1, 5, 10, and 20
    action_fns = {
        "A=1": create_action_fn(1.0),
        "A=5": create_action_fn(5.0),
        "A=10": create_action_fn(10.0),
        "A=20": create_action_fn(20.0)
    }

    # Load model
    data = np.load("models/task_3.1/model_kernel_optimized_sincos_action.npz")
    X_basis = data["X_basis"]
    alpha = data["alpha"]
    lengthscales = data["lengthscales"]
    
    out_dir = _make_output_dirs("figures/task_3.1/rollout")
    
    # Run rollouts for each action amplitude
    results = {}
    for name, action_fn in action_fns.items():
        print(f"Running rollout with {name}...")
        true_traj = rollout(x0, T=T, action_fn=action_fn, remap=True)
        model_traj = model_rollout(x0, alpha, X_basis, lengthscales, T, action_fn)
        results[name] = (true_traj, model_traj, action_fn)
        
        # Individual variable plots for this amplitude
        for i in range(4):
            plt.figure(figsize=(8, 6))
            plt.plot(true_traj[:, i], label="true", alpha=0.8, linewidth=2.0)
            plt.plot(model_traj[:, i], '--', label="model", alpha=0.8, linewidth=2.0)
            plt.xlabel("Time step")
            plt.ylabel(labels[i])
            plt.title(f"{labels[i]} evolution - Action amplitude: {name}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            fname = f"{labels[i]}_evolution_{name.replace('=', '')}"
            plt.savefig(os.path.join(out_dir["pdf"], f"{fname}.pdf"), bbox_inches='tight')
            plt.savefig(os.path.join(out_dir["png"], f"{fname}.png"), bbox_inches='tight')
            plt.close()
        
        # Combined plot with all variables for this amplitude
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axs = axs.flatten()
        
        for i in range(4):
            axs[i].plot(true_traj[:, i], label="true", alpha=0.8, linewidth=2.0)
            axs[i].plot(model_traj[:, i], '--', label="model", alpha=0.8, linewidth=2.0)
            axs[i].set_ylabel(labels[i])
            axs[i].set_title(f"{labels[i]} evolution")
            axs[i].grid(True)
            axs[i].legend()
            
        axs[-1].set_xlabel("time step")
        fig.suptitle(f"State evolution with sinusoidal action - Amplitude: {name}")
        plt.subplots_adjust(wspace=0.15, hspace=0.2, top=0.9)
        
        fname = f"combined_evolution_{name.replace('=', '')}"
        fig.savefig(os.path.join(out_dir["pdf"], f"{fname}.pdf"), bbox_inches='tight')
        fig.savefig(os.path.join(out_dir["png"], f"{fname}.png"), bbox_inches='tight')
        plt.close(fig)
        
        # Plot the action sequence for this amplitude
        plt.figure(figsize=(8, 3))
        t = np.arange(T)
        actions = np.array([action_fn(t_i, None) for t_i in t])
        plt.plot(t, actions, 'r', linewidth=2.0)
        plt.xlabel("Time step")
        plt.ylabel("Action")
        plt.title(f"Applied action sequence - Amplitude: {name}")
        plt.grid(True)
        plt.tight_layout()
        
        fname = f"action_sequence_{name.replace('=', '')}"
        plt.savefig(os.path.join(out_dir["pdf"], f"{fname}.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(out_dir["png"], f"{fname}.png"), bbox_inches='tight')
        plt.close()
    
    # Cross-comparison of different amplitudes
    for i in range(4):
        plt.figure(figsize=(10, 6))
        
        colors = ['blue', 'green', 'red', 'purple']
        for idx, (name, (true_traj, model_traj, _)) in enumerate(results.items()):
            color = colors[idx % len(colors)]
            plt.plot(true_traj[:, i], '-', color=color, label=f"true ({name})", alpha=0.7)
            plt.plot(model_traj[:, i], '--', color=color, label=f"model ({name})", alpha=0.7)
        
        plt.xlabel("Time step")
        plt.ylabel(labels[i])
        plt.title(f"{labels[i]} with different action amplitudes")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        fname = f"{labels[i]}_amplitude_comparison"
        plt.savefig(os.path.join(out_dir["pdf"], f"{fname}.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(out_dir["png"], f"{fname}.png"), bbox_inches='tight')
        plt.close()
    
    # Plot all action sequences together
    plt.figure(figsize=(10, 4))
    t = np.arange(T)
    
    colors = ['blue', 'green', 'red', 'purple']
    for idx, (name, (_, _, action_fn)) in enumerate(results.items()):
        actions = np.array([action_fn(t_i, None) for t_i in t])
        plt.plot(t, actions, color=colors[idx % len(colors)], label=name, linewidth=2.0)
    
    plt.xlabel("Time step")
    plt.ylabel("Action")
    plt.title("Action sequences comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    fname = "action_sequences_comparison"
    plt.savefig(os.path.join(out_dir["pdf"], f"{fname}.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(out_dir["png"], f"{fname}.png"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
