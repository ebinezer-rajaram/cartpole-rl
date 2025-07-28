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
    # Load the nonlinear dataset from task 4.2
    print("Loading dataset...")
    data = np.load("models/task_4.2/nonlinear_dataset.npz")
    # X = data["X"]
    # Y = data["Y"]
    
    X = np.load("data/task_3.1/X.npy")
    Y = np.load("data/task_3.1/Y.npy")
    
    # Convert to feature representation
    X_features = to_sincos_features_with_action(X)
    
    # Split the data into train and test sets
    n_samples = X.shape[0]
    train_ratio = 0.8
    train_idx = np.random.choice(n_samples, int(train_ratio * n_samples), replace=False)
    test_idx = np.setdiff1d(np.arange(n_samples), train_idx)
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    X_train_features = X_features[train_idx]
    X_test_features = X_features[test_idx]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create the basis set (randomly select subset of training examples)
    n_basis = 1000  # Adjust based on computational resources
    basis_indices = np.random.choice(len(X_train), n_basis, replace=False)
    X_basis = X_train_features[basis_indices]
    
    # Compute the kernel matrix
    print("Computing kernel matrix...")
    # Initial lengthscales - can be optimized further
    lengthscales = np.array([1.0, 1.0, 0.5, 0.5, 1.0, 1.0])
    K = rbf_kernel(X_train_features, X_basis, lengthscales)
    
    # Solve for alpha using regularized least squares
    print("Solving for model parameters...")
    lam = 1e-5  # Regularization parameter
    alpha = np.linalg.solve(K.T @ K + lam * np.eye(K.shape[1]), K.T @ Y_train)
    
    # Make predictions
    print("Making predictions...")
    Y_pred_train = predict_kernel_model(X_train_features, X_basis, alpha, lengthscales)
    Y_pred_test = predict_kernel_model(X_test_features, X_basis, alpha, lengthscales)
    
    # Calculate MSE metrics - with proper normalization
    # Compute mean squared error
    train_mse = np.mean((Y_train - Y_pred_train)**2)
    test_mse = np.mean((Y_test - Y_pred_test)**2)
    
    # Compute per-dimension MSE
    train_mse_per_dim = np.mean((Y_train - Y_pred_train)**2, axis=0)
    test_mse_per_dim = np.mean((Y_test - Y_pred_test)**2, axis=0)
    
    # Compute normalized MSE (divided by variance of target)
    train_var = np.var(Y_train, axis=0)
    test_var = np.var(Y_test, axis=0)
    
    train_nmse_per_dim = train_mse_per_dim / (train_var + 1e-10)  # avoid division by zero
    test_nmse_per_dim = test_mse_per_dim / (test_var + 1e-10)
    
    train_nmse = np.mean(train_nmse_per_dim)
    test_nmse = np.mean(test_nmse_per_dim)
    
    # Create output directory for figures
    out_dir = _make_output_dirs("figures/task_4.2/nonlinear")
    
    # Create visualization plots
    labels = ["x", "x_dot", "theta", "theta_dot"]
    print("Creating plots...")
    plot_predicted_vs_true_deltas(Y_test, Y_pred_test, labels, out_dir)
    
    # Since X has 5 columns (state + action), we need to be careful with plotting
    # Plot state variables vs. outputs
    plot_all_deltas_vs_inputs(X_test, Y_test, Y_pred_test, labels, out_dir)
    
    # Visualize some of the alpha values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(np.arange(min(20, len(alpha))), np.linalg.norm(alpha[:20], axis=1))
    ax.set_title("Alpha Vector Magnitudes (First 20)")
    ax.set_xlabel("Basis point index")
    ax.set_ylabel("||α_i||")
    plt.tight_layout()
    plt.savefig(f"{out_dir['pdf']}/alpha_magnitudes.pdf", bbox_inches='tight')
    plt.savefig(f"{out_dir['png']}/alpha_magnitudes.png", bbox_inches='tight')
    plt.close(fig)
    
    # Print results
    print("\n==== NONLINEAR MODEL RESULTS (TASK 4.2) ====")
    
    print("\nMSE comparison:")
    print(f"Train MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Train NMSE: {train_nmse:.6f}")
    print(f"Test NMSE: {test_nmse:.6f}")
    
    print("\nMSE per output dimension:")
    for i, label in enumerate(labels):
        print(f"Δ{label} - Train MSE: {train_mse_per_dim[i]:.6f} (NMSE: {train_nmse_per_dim[i]:.6f})")
        print(f"Δ{label} - Test MSE: {test_mse_per_dim[i]:.6f} (NMSE: {test_nmse_per_dim[i]:.6f})")
        print()
    
    # Save the model for further analysis
    os.makedirs("models/task_4.2", exist_ok=True)
    np.savez("models/task_4.2/nonlinear_model.npz",
            X_basis=X_basis,
            alpha=alpha,
            lengthscales=lengthscales,
            lam=lam,
            train_mse=train_mse,
            test_mse=test_mse,
            train_mse_per_dim=train_mse_per_dim,
            test_mse_per_dim=test_mse_per_dim,
            train_nmse=train_nmse,
            test_nmse=test_nmse,
            train_nmse_per_dim=train_nmse_per_dim,
            test_nmse_per_dim=test_nmse_per_dim)
    
    print("Model saved to models/task_4.2/nonlinear_model.npz")

if __name__ == "__main__":
    main()
