import numpy as np
import matplotlib.pyplot as plt
import os

from cartpole.simulation import rollout, nonlinear_model_rollout
from cartpole.kernels import fit_kernel_model
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

def main():
    labels = ["x", "x_dot", "theta", "theta_dot"]
    initial_conditions = {
        "small_theta_dot": np.array([0.0, 0.0, np.pi, 1.0]),
        "large_theta_dot": np.array([0.0, 0.0, np.pi, 10.0]),
        "large_x_dot":     np.array([0.0, 10.0, np.pi, 0.0]),
        "full_rotation":   np.array([0.0, 0.0, np.pi, 15.0])
    }

    X = np.load("data/task_1.3/X.npy")
    Y = np.load("data/task_1.3/Y.npy")
    N = X.shape[0]
    M = 100

    rng = np.random.default_rng(seed=0)
    idx = rng.choice(N, size=M, replace=False)
    basis_X = X[idx]
    lengthscales = np.std(X, axis=0)

    alpha = fit_kernel_model(X, Y, basis_X, lengthscales, lam=1e-4)

    T = 100
    out_dir = _make_output_dirs("figures/task_2.1/rollout")

    for name, x0 in initial_conditions.items():
        print(f"Simulating: {name}")
        true_traj = rollout(x0, T=T, remap=False)
        model_traj = nonlinear_model_rollout(x0, alpha, basis_X, lengthscales, T)

        # Individual variable plots
        for i in range(4):
            plt.figure(figsize=(8, 6))
            plt.plot(true_traj[:, i], label="true", alpha=0.8, linewidth=2.0)
            plt.plot(model_traj[:, i], '--', label="model", alpha=0.8, linewidth=2.0)
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
            axs[i].plot(true_traj[:, i], label="true", alpha=0.8, linewidth=2.0)
            axs[i].plot(model_traj[:, i], '--', label="model", alpha=0.8, linewidth=2.0)
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

if __name__ == "__main__":
    main()
