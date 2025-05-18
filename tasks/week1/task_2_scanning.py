import numpy as np
import matplotlib.pyplot as plt
import os
from cartpole.CartPole import CartPole

def perform_single_step(state, return_delta=False):
    env = CartPole()
    env.setState(state)
    env.performAction(0.0)
    new_state = env.getState()
    return new_state - state if return_delta else new_state

def scan_single_variable(var_index, scan_vals, base_state, return_delta=False):
    X_vals = []
    Y_outputs = []

    for val in scan_vals:
        test_state = base_state.copy()
        test_state[var_index] = val
        result = perform_single_step(test_state, return_delta)
        X_vals.append(val)
        Y_outputs.append(result)

    return np.array(X_vals), np.array(Y_outputs)

def main():
    labels = ["x", "x_dot", "theta", "theta_dot"]

    # TOGGLE HERE
    return_delta = True  # False → Y = X(T), True → Y = X(T) - X(0)

    scan_ranges = {
        0: np.linspace(-5, 5, 100),
        1: np.linspace(-10, 10, 100),
        2: np.linspace(-np.pi, np.pi, 100),
        3: np.linspace(-15, 15, 100),
    }

    base_state = np.random.uniform(
        low=[-5, -10, -np.pi, -15],
        high=[5, 10, np.pi, 15],
        size=4
    )

    tag = "delta" if return_delta else "next"
    base_dir = f"figures/task_1.2/{tag}"
    pdf_dir = os.path.join(base_dir, "pdf")
    png_dir = os.path.join(base_dir, "png")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    for i in range(4):
        X_vals, Y_vals = scan_single_variable(i, scan_ranges[i], base_state, return_delta)

        for j in range(4):
            plt.figure()
            plt.plot(X_vals, Y_vals[:, j])
            plt.xlabel(labels[i])
            ylabel = f"Δ{labels[j]}" if return_delta else f"{labels[j]} (next)"
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} vs {labels[i]}")
            plt.grid(True)

            fname = f"{ylabel.replace(' ', '_')}_vs_{labels[i]}"
            plt.savefig(os.path.join(pdf_dir, f"{fname}.pdf"))
            plt.savefig(os.path.join(png_dir, f"{fname}.png"))
            plt.show()
            plt.close()

if __name__ == "__main__":
    main()
