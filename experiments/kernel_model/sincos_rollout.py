import numpy as np
import matplotlib.pyplot as plt
import os

from cartpole.simulation import rollout
from cartpole.CartPole import remap_angle
from cartpole.plotting import _make_output_dirs
from .sincos_jax_regression import predict_kernel

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

def to_sincos_features(x):
    return np.array([
        x[0],
        x[1],
        np.sin(x[2]),
        np.cos(x[2]),
        x[3]
    ])

def model_rollout_sincos(x0, alpha, basis_X, lengthscales, T):
    X = np.zeros((T, 4))
    x = x0.copy()
    for t in range(T):
        X[t] = x
        x_feat = to_sincos_features(x)[None, :]
        dx = predict_kernel(x_feat, basis_X, alpha, lengthscales)[0]
        x = np.array(x + dx)  # force NumPy for mutable update
        x[2] = remap_angle(x[2])
    return X

def count_oscillations(theta_series):
    unwrapped = np.unwrap(theta_series)
    crossings = np.diff(np.signbit(np.sin(unwrapped)))
    return int(np.abs(np.sum(crossings)) // 2)

def time_to_deviation(traj_true, traj_model, threshold=0.5):
    diffs = np.linalg.norm(traj_true - traj_model, axis=1)
    above = np.where(diffs > threshold)[0]
    return above[0] if len(above) > 0 else len(diffs)

def main():
    labels = ["x", "x_dot", "theta", "theta_dot"]
    initial_conditions = {
        "small_theta_dot": np.array([0.0, 0.0, np.pi, 1.0]),
        "large_theta_dot": np.array([0.0, 0.0, np.pi, 10.0]),
        "large_x_dot":     np.array([0.0, 10.0, np.pi, 0.0]),
        "full_rotation":   np.array([0.0, 0.0, np.pi, 15.0])
    }

    data = np.load("models/kernel_model/model_kernel_optimized_sincos.npz")
    X_basis = data["X_basis"]
    alpha = data["alpha"]
    lengthscales = data["lengthscales"]

    T = 100
    out_dir = _make_output_dirs("figures/kernel_model/sincos/rollout")

    for name, x0 in initial_conditions.items():
        print(f"Simulating: {name}")
        traj_true = rollout(x0, T=T, remap=True)
        traj_model = model_rollout_sincos(x0, alpha, X_basis, lengthscales, T)

        # Individual variable plots
        for i in range(4):
            plt.figure(figsize=(8, 6))
            plt.plot(traj_true[:, i], label="true", alpha=0.8, linewidth=2.0)
            plt.plot(traj_model[:, i], '--', label="model", alpha=0.8, linewidth=2.0)
            plt.xlabel("Time step")
            plt.ylabel(labels[i])
            plt.title(f"{labels[i]} over time — {name}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            fname = f"{labels[i]}_evolution_{name}"
            plt.savefig(os.path.join(out_dir["pdf"], f"{fname}.pdf"), bbox_inches='tight')
            plt.savefig(os.path.join(out_dir["png"], f"{fname}.png"), bbox_inches='tight')
            plt.close()
        
        # Combined plot with all variables
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axs = axs.flatten()
        
        for i in range(4):
            axs[i].plot(traj_true[:, i], label="true", alpha=0.8, linewidth=2.0)
            axs[i].plot(traj_model[:, i], '--', label="model", alpha=0.8, linewidth=2.0)
            axs[i].set_ylabel(labels[i])
            axs[i].set_title(f"{labels[i]} evolution")
            axs[i].grid(True)
            axs[i].legend()
            
        axs[-1].set_xlabel("time step")
        fig.suptitle(f"State evolution — {name}")
        plt.subplots_adjust(wspace=0.15, hspace=0.2, top=0.9)
        
        fname = f"combined_evolution_{name}"
        fig.savefig(os.path.join(out_dir["pdf"], f"{fname}.pdf"), bbox_inches='tight')
        fig.savefig(os.path.join(out_dir["png"], f"{fname}.png"), bbox_inches='tight')
        plt.close(fig)

        t_dev = time_to_deviation(traj_true, traj_model)
        cycles = count_oscillations(traj_model[:, 2])
        print(f"  → time to deviation (>0.5): {t_dev} steps")
        print(f"  → oscillation cycles: {cycles}")

if __name__ == "__main__":
    main()
