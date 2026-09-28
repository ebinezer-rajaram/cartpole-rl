import numpy as np
from cartpole.scanning import scan_model_vs_true_action, scan2d_model_vs_true_action
from cartpole.plotting import _make_output_dirs, plot_scan_comparison, plot_2d_slices
from .action_model import predict_kernel

model = np.load("models/policy_search/model_kernel_optimized_sincos_action.npz")
X_basis = model["X_basis"]
alpha = model["alpha"]
lengthscales = model["lengthscales"]

def model_predict_fn(state_action):
    x, x_dot, theta, theta_dot, action = state_action
    features = np.array([
        x,
        x_dot,
        np.sin(theta),
        np.cos(theta),
        theta_dot,
        action
    ])[None, :]
    return predict_kernel(features, X_basis, alpha, lengthscales)[0]

def main():
    labels = ["x", "x_dot", "theta", "theta_dot", "action"]
    output_labels = ["x", "x_dot", "theta", "theta_dot"]
    out_dir = _make_output_dirs("figures/policy_search/action_model/scans")

    base_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # 4D state + 1D action
    scan_ranges = {
        0: (-2, 2),        # x
        1: (-5, 5),        # x_dot
        2: (-np.pi, np.pi),# theta
        3: (-10, 10),      # theta_dot
        4: (-20, 20)       # action
    }

    for i in range(5):
        scan_vals = np.linspace(*scan_ranges[i], 100)
        X_vals, Y_true, Y_pred = scan_model_vs_true_action(i, scan_vals, base_state, model_predict_fn)
        plot_scan_comparison(scan_vals, Y_true, Y_pred, labels[i], output_labels, out_dir, tag="model")

    pairs = [(0, 4), (1, 4), (2, 4), (3, 4)]
    for i, j in pairs:
        X_coords, Y_true, Y_pred = scan2d_model_vs_true_action(
            i, j, scan_ranges[i], scan_ranges[j], base_state, model_predict_fn
        )
        for k in range(4):
            plot_2d_slices(X_coords, Y_true[:, k], Y_pred[:, k],
                           labels[i], labels[j], f"Δ{output_labels[k]}",
                           out_dir, tag="model")

    print("Scan plots saved to figures/policy_search/action_model/scans/")

if __name__ == "__main__":
    main()
