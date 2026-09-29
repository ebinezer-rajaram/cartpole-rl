import numpy as np
import matplotlib.pyplot as plt
import os

from cartpole.plotting import (
    _make_output_dirs,
    plot_predicted_vs_true_deltas,
    plot_all_deltas_vs_inputs
)

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
    out_dir = _make_output_dirs("figures/robustness/process_noise/linear")
    
    # Reference process-noise dataset written by noisy_data.py
    data = np.load("models/robustness/process_noise/nonlinear_dataset.npz")
    X = data["X"]
    Y = data["Y"]
    
    n_samples = X.shape[0]
    train_ratio = 0.8
    train_idx = np.random.choice(n_samples, int(train_ratio * n_samples), replace=False)
    test_idx = np.setdiff1d(np.arange(n_samples), train_idx)
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    
    C = np.linalg.lstsq(X_train, Y_train, rcond=None)[0].T
    
    Y_pred_train = X_train @ C.T
    Y_pred_test = X_test @ C.T

    train_mse = np.mean((Y_train - Y_pred_train)**2)
    test_mse = np.mean((Y_test - Y_pred_test)**2)
    
    train_mse_per_dim = np.mean((Y_train - Y_pred_train)**2, axis=0)
    test_mse_per_dim = np.mean((Y_test - Y_pred_test)**2, axis=0)
    
    plot_predicted_vs_true_deltas(Y_test, Y_pred_test, labels, out_dir)
    plot_all_deltas_vs_inputs(X_test, Y_test, Y_pred_test, labels, out_dir)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(C, cmap='viridis')
    ax.set_title("Linear Model Matrix")
    ax.set_xlabel("Input dimension")
    ax.set_ylabel("Output dimension")
    ax.set_xticks(np.arange(C.shape[1]))
    ax.set_yticks(np.arange(C.shape[0]))
    ax.set_xticklabels(labels + ["action"])  # inputs are state + action
    ax.set_yticklabels([f"Δ{label}" for label in labels])
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f"{out_dir['pdf']}/model_matrix.pdf", bbox_inches='tight')
    plt.savefig(f"{out_dir['png']}/model_matrix.png", bbox_inches='tight')
    plt.close(fig)
    
    print("\n==== LINEAR MODEL RESULTS (PROCESS NOISE) ====")
    print("\nModel matrix:")
    print(np.round(C, 4))
    
    print("\nMSE comparison:")
    print(f"Train MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    
    print("\nMSE per output dimension:")
    for i, label in enumerate(labels):
        print(f"Δ{label} - Train MSE: {train_mse_per_dim[i]:.6f}")
        print(f"Δ{label} - Test MSE: {test_mse_per_dim[i]:.6f}")
        print()
        
    os.makedirs("models/robustness/process_noise", exist_ok=True)
    np.savez("models/robustness/process_noise/linear_model.npz",
            C=C,
            train_mse=train_mse,
            test_mse=test_mse,
            train_mse_per_dim=train_mse_per_dim,
            test_mse_per_dim=test_mse_per_dim)
    
    print("Model saved to models/robustness/process_noise/linear_model.npz")

if __name__ == "__main__":
    main()
