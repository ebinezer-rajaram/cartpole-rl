import numpy as np
import matplotlib.pyplot as plt
import os

from cartpole.simulation import rollout, model_rollout

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

def main():
    labels = ["x", "x_dot", "theta", "theta_dot"]
    initial_conditions = {
        "small_theta_dot": np.array([0.0, 0.0, np.pi, 1.0]),
        "large_theta_dot": np.array([0.0, 0.0, np.pi, 10.0]),
        "large_x_dot":     np.array([0.0, 10.0, np.pi, 0.0]),
        "full_rotation":   np.array([0.0, 0.0, np.pi, 15.0])
    }
    
    X = np.load("data/state_transitions/X.npy")
    Y = np.load("data/state_transitions/Y.npy")
    C = np.linalg.lstsq(X, Y, rcond=None)[0].T

    T = 200
    pdf_dir = "figures/linear_model/rollout_compare/pdf"
    png_dir = "figures/linear_model/rollout_compare/png"
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    for name, init_state in initial_conditions.items():
        print(f"Simulating: {name}")
        traj_true = rollout(init_state, T=T, remap=False)
        traj_model = model_rollout(init_state, C, T=T, remap_theta=True)

        for i in range(4):
            plt.figure(figsize=(8, 6))
            plt.plot(traj_true[:, i], label="true", alpha=0.8, linewidth=2.0)
            plt.plot(traj_model[:, i], label="model", linestyle='--', alpha=0.8, linewidth=2.0)
            plt.xlabel("time step")
            plt.ylabel(labels[i])
            plt.title(f"{labels[i]} over time — {name}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            fname = f"{labels[i]}_evolution_{name}"
            plt.savefig(f"{pdf_dir}/{fname}.pdf", bbox_inches='tight')
            plt.savefig(f"{png_dir}/{fname}.png", bbox_inches='tight')
            plt.close()

        # Combined plot with all variables
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axs = axs.flatten()
        
        for i in range(4):
            axs[i].plot(traj_true[:, i], label="true", alpha=0.8, linewidth=2.0)
            axs[i].plot(traj_model[:, i], label="model", linestyle='--', alpha=0.8, linewidth=2.0)
            axs[i].set_ylabel(labels[i])
            axs[i].set_title(f"{labels[i]} evolution")
            axs[i].grid(True)
            axs[i].legend()
            
        axs[-1].set_xlabel("time step")
        fig.suptitle(f"State evolution — {name}")
        plt.subplots_adjust(wspace=0.15, hspace=0.2, top=0.9)
        
        fname = f"combined_evolution_{name}"
        fig.savefig(f"{pdf_dir}/{fname}.pdf", bbox_inches='tight')
        fig.savefig(f"{png_dir}/{fname}.png", bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    main()
