import numpy as np
import matplotlib.pyplot as plt
import os

from cartpole.simulation import rollout
from cartpole.CartPole import remap_angle
from .regression import predict_kernel
from cartpole.plotting import _make_output_dirs

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
    T = 20

    def action_fn(t, state):
        return 5.0 * np.sin(0.05 * t)

    # Load model
    data = np.load("models/task_3.1/model_kernel_optimized_sincos_action.npz")
    X_basis = data["X_basis"]
    alpha = data["alpha"]
    lengthscales = data["lengthscales"]

    true_traj = rollout(x0, T=T, action_fn=action_fn, remap=True)
    model_traj = model_rollout(x0, alpha, X_basis, lengthscales, T, action_fn)

    out_dir = _make_output_dirs("figures/task_3.1/rollout")

    for i in range(4):
        plt.figure()
        plt.plot(true_traj[:, i], label="true", alpha=0.8)
        plt.plot(model_traj[:, i], '--', label="model", alpha=0.8)
        plt.xlabel("Time step")
        plt.ylabel(labels[i])
        plt.title(f"{labels[i]} over time")
        plt.legend()
        plt.grid(True)

        fname = f"{labels[i]}_evolution"
        plt.savefig(os.path.join(out_dir["pdf"], f"{fname}.pdf"))
        plt.savefig(os.path.join(out_dir["png"], f"{fname}.png"))
        plt.close()

if __name__ == "__main__":
    main()
