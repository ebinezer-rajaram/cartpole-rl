import numpy as np
import os
from cartpole.simulation import rollout
from cartpole.plotting import plot_time_series, plot_phase_portraits

def main():
    save_dir = "figures/linear_model/simulation"
    data_dir = "data/linear_model/simulation"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    scenarios = {
        "oscillation_small_theta_dot": np.array([0.0, 0.0, np.pi, 1.0]),
        "oscillation_large_theta_dot": np.array([0.0, 0.0, np.pi, 10.0]),
        "oscillation_small_x_dot": np.array([0.0, 1.0, np.pi, 0.0]),
        "rotation_theta_dot": np.array([0.0, 0.0, np.pi, 15.0]),
        "rotation_x_dot": np.array([0.0, 10.0, np.pi, 0.0]),
    }

    for name, init_state in scenarios.items():
        print(f"Simulating: {name}")
        traj = rollout(init_state, T=200, action_fn=None, remap=True, visual=False)

        np.save(os.path.join(data_dir, f"{name}.npy"), traj)
        plot_time_series(traj, name, save_dir)
        plot_phase_portraits(traj, name, save_dir)

if __name__ == "__main__":
    main()
