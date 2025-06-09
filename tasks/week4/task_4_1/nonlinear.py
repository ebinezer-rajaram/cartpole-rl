import numpy as np
import matplotlib.pyplot as plt
import os

from cartpole.plotting import (
    _make_output_dirs,
    plot_predicted_vs_true_deltas,
    plot_all_deltas_vs_inputs
)

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

def to_sincos_features_with_action(X):
    """Transform state-action pairs to feature representation with sin/cos for angles."""
    X_new = np.zeros((X.shape[0], 6))
    X_new[:, 0] = X[:, 0]  # x
    X_new[:, 1] = X[:, 1]  # x_dot
    X_new[:, 2] = np.sin(X[:, 2])  # sin(theta)
    X_new[:, 3] = np.cos(X[:, 2])  # cos(theta)
    X_new[:, 4] = X[:, 3]  # theta_dot
    X_new[:, 5] = X[:, 4]  # action
    return X_new

def rbf_kernel(X1, X2, lengthscales):
    """Compute RBF kernel matrix between X1 and X2."""
    sq_dist = np.sum(((X1[:, None, :] - X2[None, :, :]) / lengthscales[None, None, :])**2, axis=2)
    return np.exp(-0.5 * sq_dist)

def predict_kernel_model(X_new, basis_X, alpha, lengthscales):
    """Predict using trained kernel model."""
    K = rbf_kernel(X_new, basis_X, lengthscales)
    return K @ alpha

def main():
    # Load the clean and noisy kernel models
    clean_model = np.load("models/task_3.1/model_kernel_optimized_sincos_action.npz")
    
    try:
        noisy_model = np.load("models/task_4.1/nonlinear_models.npz")
        noisy_model_exists = True
    except:
        # If noisy model doesn't exist yet, we'll create it
        noisy_model_exists = False
    
    # Load validation datasets
    print("Loading datasets...")
    X_clean = np.load("data/task_3.1/X.npy")
    Y_clean = np.load("data/task_3.1/Y.npy")
    
    noisy_data = np.load("data/task_4.1/noisy_nonlinear_dataset.npz")
    X_noisy = noisy_data["X"]
    Y_noisy = noisy_data["Y"]
    
    # Convert to feature representation
    X_clean_features = to_sincos_features_with_action(X_clean)
    X_noisy_features = to_sincos_features_with_action(X_noisy)
    
    # Get model parameters
    clean_X_basis = clean_model["X_basis"]
    clean_alpha = clean_model["alpha"]
    clean_lengthscales = clean_model["lengthscales"]
    
    # If noisy model doesn't exist yet, create it from scratch
    if not noisy_model_exists:
        print("Noisy model not found. Creating new model...")
        
        # Compute kernel matrix for noisy data
        K = rbf_kernel(X_noisy_features, clean_X_basis, clean_lengthscales)
        
        # Solve for alpha using regularized least squares
        lam = clean_model["lam"]
        noisy_alpha = np.linalg.solve(K.T @ K + lam * np.eye(K.shape[1]), K.T @ Y_noisy)
        
        # Save the noisy model
        os.makedirs("models/task_4.1", exist_ok=True)
        np.savez("models/task_4.1/nonlinear_models.npz",
                clean_X_basis=clean_X_basis,
                clean_alpha=clean_alpha,
                clean_lengthscales=clean_lengthscales,
                noisy_alpha=noisy_alpha)
        
        print("Noisy model saved to models/task_4.1/nonlinear_models.npz")
    else:
        # Load the noisy model
        noisy_alpha = noisy_model["noisy_alpha"]
    
    # Create output directory for figures
    out_dir = _make_output_dirs("figures/task_4.1/nonlinear_noise")
    
    # Make predictions with both models
    print("Making predictions...")
    Y_clean_pred_clean = predict_kernel_model(X_clean_features, clean_X_basis, clean_alpha, clean_lengthscales)
    Y_noisy_pred_noisy = predict_kernel_model(X_noisy_features, clean_X_basis, noisy_alpha, clean_lengthscales)
    Y_clean_pred_noisy = predict_kernel_model(X_clean_features, clean_X_basis, noisy_alpha, clean_lengthscales)
    Y_noisy_pred_clean = predict_kernel_model(X_noisy_features, clean_X_basis, clean_alpha, clean_lengthscales)
    
    # Calculate MSE metrics
    clean_mse_clean_model = np.mean((Y_clean - Y_clean_pred_clean)**2)
    clean_mse_noisy_model = np.mean((Y_clean - Y_clean_pred_noisy)**2)
    noisy_mse_clean_model = np.mean((Y_noisy - Y_noisy_pred_clean)**2)
    noisy_mse_noisy_model = np.mean((Y_noisy - Y_noisy_pred_noisy)**2)
    
    # Calculate per-dimension MSE for more detailed analysis
    clean_mse_clean_model_per_dim = np.mean((Y_clean - Y_clean_pred_clean)**2, axis=0)
    clean_mse_noisy_model_per_dim = np.mean((Y_clean - Y_clean_pred_noisy)**2, axis=0)
    noisy_mse_noisy_model_per_dim = np.mean((Y_noisy - Y_noisy_pred_noisy)**2, axis=0)
    
    # Create visualization plots
    labels = ["x", "x_dot", "theta", "theta_dot"]
    print("Creating plots...")
    # Fix function calls to match the correct signatures in plotting.py
    plot_predicted_vs_true_deltas(Y_clean, Y_clean_pred_clean, labels, out_dir)
    plot_predicted_vs_true_deltas(Y_noisy, Y_noisy_pred_noisy, labels, out_dir)
    plot_predicted_vs_true_deltas(Y_clean, Y_clean_pred_noisy, labels, out_dir)
    
    plot_all_deltas_vs_inputs(X_clean[:, :4], Y_clean, Y_clean_pred_clean, labels, out_dir)
    plot_all_deltas_vs_inputs(X_noisy[:, :4], Y_noisy, Y_noisy_pred_noisy, labels, out_dir)
    
    # Add plot for noisy model on clean data
    plot_all_deltas_vs_inputs(X_clean[:, :4], Y_clean, Y_clean_pred_noisy, labels, out_dir)
    
    # Visualize alpha differences
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    alpha_diff_magnitude = np.linalg.norm(clean_alpha - noisy_alpha, axis=1)
    
    axs[0].hist(alpha_diff_magnitude, bins=30)
    axs[0].set_title("Distribution of Differences in Alpha Vectors")
    axs[0].set_xlabel("||α_clean - α_noisy||")
    axs[0].set_ylabel("Count")
    
    axs[1].scatter(np.arange(len(alpha_diff_magnitude)), alpha_diff_magnitude, alpha=0.5)
    axs[1].set_title("Alpha Vector Differences")
    axs[1].set_xlabel("Basis point index")
    axs[1].set_ylabel("||α_clean - α_noisy||")
    
    plt.tight_layout()
    plt.savefig(f"{out_dir['pdf']}/alpha_differences.pdf", bbox_inches='tight')
    plt.savefig(f"{out_dir['png']}/alpha_differences.png", bbox_inches='tight')
    plt.close(fig)
    
    # Print analysis results
    print("\n==== NONLINEAR MODEL COMPARISON ====")
    
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
    
    # Analyze clean vs. noisy model differences
    l2_alpha_diff = np.linalg.norm(clean_alpha - noisy_alpha)
    l2_alpha_clean = np.linalg.norm(clean_alpha)
    relative_diff_pct = 100 * l2_alpha_diff / l2_alpha_clean
    
    print("\nModel parameter differences:")
    print(f"L2 difference between alpha vectors: {l2_alpha_diff:.4f}")
    print(f"Relative difference: {relative_diff_pct:.2f}%")
    
    # Calculate percentage degradation due to noise
    degradation_pct = 100 * (clean_mse_noisy_model - clean_mse_clean_model) / clean_mse_clean_model
    print(f"\nPerformance degradation on clean data: {degradation_pct:.2f}%")

if __name__ == "__main__":
    main()