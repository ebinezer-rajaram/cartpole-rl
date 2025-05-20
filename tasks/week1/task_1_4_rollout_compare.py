import numpy as np
import matplotlib.pyplot as plt
import os

from cartpole.simulation import rollout, model_rollout

def main():
    labels = ["x", "x_dot", "theta", "theta_dot"]
    initial_conditions = {
        "small_theta_dot": np.array([0.0, 0.0, np.pi, 1.0]),
        "large_theta_dot": np.array([0.0, 0.0, np.pi, 10.0]),
        "large_x_dot":     np.array([0.0, 10.0, np.pi, 0.0]),
        "full_rotation":   np.array([0.0, 0.0, np.pi, 15.0])
    }

    # Load trained model
    X = np.load("data/task_1.3/X.npy")
    Y = np.load("data/task_1.3/Y.npy")
    C = np.linalg.lstsq(X, Y, rcond=None)[0].T

    T = 200
    pdf_dir = "figures/task_1.4/pdf"
    png_dir = "figures/task_1.4/png"
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    for name, init_state in initial_conditions.items():
        print(f"Simulating: {name}")
        traj_true = rollout(init_state, T=T, remap=False)
        traj_model = model_rollout(init_state, C, T=T, remap_theta=True)

        for i in range(4):
            plt.figure()
            plt.plot(traj_true[:, i], label="true", alpha=0.8)
            plt.plot(traj_model[:, i], label="model", linestyle='--', alpha=0.8)
            plt.xlabel("time step")
            plt.ylabel(labels[i])
            plt.title(f"{labels[i]} over time — {name}")
            plt.legend()
            plt.grid(True)

            fname = f"{labels[i]}_evolution_{name}"
            plt.savefig(f"{pdf_dir}/{fname}.pdf")
            plt.savefig(f"{png_dir}/{fname}.png")
            plt.close()

if __name__ == "__main__":
    main()
