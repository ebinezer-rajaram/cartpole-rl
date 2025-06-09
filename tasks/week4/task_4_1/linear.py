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
    out_dir = _make_output_dirs("figures/task_4.1/linear_noise")
    
    X_clean = np.load("data/task_1.3/X.npy")
    Y_clean = np.load("data/task_1.3/Y.npy")
    
    noisy_data = np.load("data/task_4.1/noisy_dataset.npz")
    X_noisy = noisy_data["X"]
    Y_noisy = noisy_data["Y"]
    
    C_clean = np.linalg.lstsq(X_clean, Y_clean, rcond=None)[0].T
    C_noisy = np.linalg.lstsq(X_noisy, Y_noisy, rcond=None)[0].T
    
    Y_pred_clean = X_clean @ C_clean.T
    Y_pred_noisy = X_noisy @ C_noisy.T
    
    clean_mse_clean_model = np.mean((Y_clean - Y_pred_clean)**2)
    clean_mse_noisy_model = np.mean((Y_clean - (X_clean @ C_noisy.T))**2)
    
    noisy_mse_clean_model = np.mean((Y_noisy - (X_noisy @ C_clean.T))**2)
    noisy_mse_noisy_model = np.mean((Y_noisy - Y_pred_noisy)**2)
    
    clean_mse_clean_model_per_dim = np.mean((Y_clean - Y_pred_clean)**2, axis=0)
    clean_mse_noisy_model_per_dim = np.mean((Y_clean - (X_clean @ C_noisy.T))**2, axis=0)
    noisy_mse_noisy_model_per_dim = np.mean((Y_noisy - Y_pred_noisy)**2, axis=0)
    
    # Fix function calls to match signatures in plotting.py
    plot_predicted_vs_true_deltas(Y_clean, Y_pred_clean, labels, out_dir)
    plot_predicted_vs_true_deltas(Y_noisy, Y_pred_noisy, labels, out_dir)
    plot_predicted_vs_true_deltas(Y_clean, X_clean @ C_noisy.T, labels, out_dir)
    
    plot_all_deltas_vs_inputs(X_clean, Y_clean, Y_pred_clean, labels, out_dir)
    plot_all_deltas_vs_inputs(X_noisy, Y_noisy, Y_pred_noisy, labels, out_dir)
    
    # Compare model matrices
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    im0 = axs[0].imshow(C_clean, cmap='viridis')
    axs[0].set_title("Clean Model Matrix")
    axs[0].set_xlabel("Input dimension")
    axs[0].set_ylabel("Output dimension")
    axs[0].set_xticks(np.arange(C_clean.shape[1]))
    axs[0].set_yticks(np.arange(C_clean.shape[0]))
    axs[0].set_xticklabels(labels)
    axs[0].set_yticklabels([f"Δ{label}" for label in labels])
    plt.colorbar(im0, ax=axs[0])
    
    im1 = axs[1].imshow(C_noisy, cmap='viridis')
    axs[1].set_title("Noisy Model Matrix")
    axs[1].set_xlabel("Input dimension")
    axs[1].set_xticks(np.arange(C_noisy.shape[1]))
    axs[1].set_yticks(np.arange(C_noisy.shape[0]))
    axs[1].set_xticklabels(labels)
    axs[1].set_yticklabels([f"Δ{label}" for label in labels])
    plt.colorbar(im1, ax=axs[1])
    
    plt.tight_layout()
    plt.savefig(f"{out_dir['pdf']}/model_matrix_comparison.pdf", bbox_inches='tight')
    plt.savefig(f"{out_dir['png']}/model_matrix_comparison.png", bbox_inches='tight')
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(C_noisy - C_clean, cmap='coolwarm')
    ax.set_title("Model Matrix Difference (Noisy - Clean)")
    ax.set_xlabel("Input dimension")
    ax.set_ylabel("Output dimension")
    ax.set_xticks(np.arange(C_clean.shape[1]))
    ax.set_yticks(np.arange(C_clean.shape[0]))
    ax.set_xticklabels(labels)
    ax.set_yticklabels([f"Δ{label}" for label in labels])
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f"{out_dir['pdf']}/model_matrix_difference.pdf", bbox_inches='tight')
    plt.savefig(f"{out_dir['png']}/model_matrix_difference.png", bbox_inches='tight')
    plt.close(fig)
    
    print("\n==== LINEAR MODEL COMPARISON ====")
    print("\nModel matrices:")
    print("Clean model:\n", np.round(C_clean, 4))
    print("Noisy model:\n", np.round(C_noisy, 4))
    print("Absolute difference:\n", np.round(np.abs(C_clean - C_noisy), 4))
    print("Relative difference (%):\n", np.round(100 * np.abs(C_clean - C_noisy) / (np.abs(C_clean) + 1e-10), 2))
    
    print("\nMSE comparison:")
    print(f"Clean model on clean data: {clean_mse_clean_model:.6f}")
    print(f"Noisy model on noisy data: {noisy_mse_noisy_model:.6f}")
    print(f"Noisy model on clean data: {clean_mse_noisy_model:.6f}")
    print(f"Clean model on noisy data: {noisy_mse_clean_model:.6f}")
    
    print("\nMSE per output dimension:")
    for i, label in enumerate(labels):
        print(f"Δ{label} - Clean model on clean data: {clean_mse_clean_model_per_dim[i]:.6f}")
        print(f"Δ{label} - Noisy model on noisy data: {noisy_mse_noisy_model_per_dim[i]:.6f}")
        print(f"Δ{label} - Noisy model on clean data: {clean_mse_noisy_model_per_dim[i]:.6f}")
        print()
        
    # Save the clean and noisy models for further analysis
    os.makedirs("models/task_4.1", exist_ok=True)
    np.savez("models/task_4.1/linear_models.npz",
            C_clean=C_clean,
            C_noisy=C_noisy,
            clean_mse=clean_mse_clean_model,
            noisy_mse=noisy_mse_noisy_model,
            cross_mse_clean=clean_mse_noisy_model,
            cross_mse_noisy=noisy_mse_clean_model)
    
    print("Models saved to models/task_4.1/linear_models.npz")

if __name__ == "__main__":
    main()