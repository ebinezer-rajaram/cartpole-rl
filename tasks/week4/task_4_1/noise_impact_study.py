import numpy as np
import matplotlib.pyplot as plt
import os
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
import time

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

# ----- Functions for data processing (based on data.py) -----
def add_noise_to_dataset(X, Y, noise_std):
    """Add Gaussian noise to the dataset."""
    Y_noisy = Y + np.random.normal(0, noise_std, size=Y.shape)
    return X, Y_noisy

# ----- Functions for feature transformation (based on nonlinear.py) -----
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

def train_kernel_model(X_train, Y_train, X_features, n_basis=1000, lam=1e-5):
    """Train a kernel model with RBF basis."""
    # Create the basis set (randomly select subset of training examples)
    basis_indices = np.random.choice(len(X_train), n_basis, replace=False)
    X_basis = X_features[basis_indices]
    
    # Initial lengthscales - can be optimized further
    lengthscales = np.array([1.0, 1.0, 0.5, 0.5, 1.0, 1.0])
    
    # Compute the kernel matrix
    K = rbf_kernel(X_features, X_basis, lengthscales)
    
    # Solve for alpha using regularized least squares
    alpha = np.linalg.solve(K.T @ K + lam * np.eye(K.shape[1]), K.T @ Y_train)
    
    return X_basis, alpha, lengthscales

# ----- Policy Optimization Functions (based on policy.py) -----
def rbf_kernel_jax(X1, X2, lengthscales):
    def kernel_fn(x, y):
        scaled = (x - y) / lengthscales
        return jnp.exp(-0.5 * jnp.sum(scaled**2))
    return jax.vmap(lambda x: jax.vmap(lambda y: kernel_fn(x, y))(X2))(X1)

@jax.jit
def predict_kernel(X_test, X_basis, alpha, lengthscales):
    K = rbf_kernel_jax(X_test, X_basis, lengthscales)
    return K @ alpha

def build_features(x, a):
    return jnp.array([
        x[0],
        x[1],
        jnp.sin(x[2]),
        jnp.cos(x[2]),
        x[3],
        a
    ])[None, :]

def model_rollout_jax(x0, policy, T, X_basis, alpha, lengthscales, max_force=20.0):
    """Run model rollout using learned dynamics."""
    def step(x, _):
        a = jnp.dot(policy, x)
        a = jnp.clip(a, -max_force, max_force)
        features = build_features(x, a)
        dx = predict_kernel(features, X_basis, alpha, lengthscales)[0]
        x_next = x + dx
        # Remap angle
        x_next = x_next.at[2].set(((x_next[2] + jnp.pi) % (2 * jnp.pi)) - jnp.pi)
        return x_next, x_next
    _, traj = jax.lax.scan(step, x0, None, length=T)
    return traj

def trajectory_loss_jax(traj, X0=None, sigma_l=0.5):
    """Calculate loss for trajectory."""
    if X0 is None:
        X0 = jnp.zeros(4)
    sigma_l = jnp.asarray(sigma_l)
    diffs = (traj - X0) / sigma_l
    sq_norms = jnp.sum(diffs**2, axis=-1)
    losses = 1 - jnp.exp(-0.5 * sq_norms)
    return jnp.sum(losses)

def optimize_policy(x0, X_basis, alpha, lengthscales, T=20, max_force=20.0, sigma_l=None):
    """Optimize a linear policy for the model."""
    if sigma_l is None:
        sigma_l = jnp.array([0.5, 0.4, 0.05, 0.25])
    
    p0 = np.zeros(4)  # initial policy
    
    def objective(p):
        traj = model_rollout_jax(x0, p, T, X_basis, alpha, lengthscales, max_force=max_force)
        return trajectory_loss_jax(traj, X0=jnp.zeros(4), sigma_l=sigma_l)

    obj_and_grad = jax.jit(jax.value_and_grad(objective))

    res = minimize(
        lambda p: float(obj_and_grad(p)[0]),
        p0,
        jac=lambda p: np.array(obj_and_grad(p)[1]),
        method="L-BFGS-B",
        options={"maxiter": 100}
    )
    
    return res.x, res.fun

def evaluate_with_true_system(x0, policy, T=100):
    """Evaluate policy performance on true dynamics."""
    from cartpole.simulation import rollout
    
    max_force = 20.0
    
    def linear_policy_fn(t, state):
        force = np.dot(policy, state)
        return np.clip(force, -max_force, max_force)
    
    traj = rollout(np.array(x0), T=T, action_fn=linear_policy_fn, remap=True)
    
    # Calculate true system loss
    sigma_l = np.array([0.5, 0.4, 0.05, 0.25])
    diffs = (traj - np.zeros((1, 4))) / sigma_l
    sq_norms = np.sum(diffs**2, axis=1)
    losses = 1 - np.exp(-0.5 * sq_norms)
    true_loss = np.sum(losses)
    
    return traj, true_loss

def main():
    # Create output directories
    os.makedirs("models/noise_impact", exist_ok=True)
    os.makedirs("figures/noise_impact", exist_ok=True)
    
    # Load the clean dataset
    print("Loading clean dataset...")
    try:
        X = np.load("data/task_3.1/X.npy")
        Y = np.load("data/task_3.1/Y.npy")
    except FileNotFoundError:
        print("Error: Could not find task 3.1 dataset. Please run task 3.1 data generation first.")
        return
    
    # Configuration parameters
    np.random.seed(42)
    noise_levels = [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5]
    n_basis = 1000
    lam = 1e-5
    policy_T = 20  # time steps for policy optimization
    eval_T = 20    # longer horizon for evaluation
    
    # Initial conditions for policy testing
    x0_list = [
        ("upright", jnp.array([0.0, 0.0, 0.1, 0.0])),
        # ("tilted", jnp.array([0.0, 0.0, 0.3, 0.0])),
        # ("velocity", jnp.array([0.0, 0.2, 0.1, 0.1])),
    ]
    
    # Feature transformation (no noise dependence)
    print("Creating feature representation...")
    X_features = to_sincos_features_with_action(X)
    
    # Split data into train and test sets
    n_samples = X.shape[0]
    train_ratio = 0.8
    train_idx = np.random.choice(n_samples, int(train_ratio * n_samples), replace=False)
    test_idx = np.setdiff1d(np.arange(n_samples), train_idx)
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    X_train_features = X_features[train_idx]
    X_test_features = X_features[test_idx]
    
    # Track results
    results = {
        "noise_levels": noise_levels,
        "model_metrics": [],
        "policy_metrics": {scenario: [] for scenario, _ in x0_list},
        "true_system_metrics": {scenario: [] for scenario, _ in x0_list}
    }
    
    # For each noise level
    for noise_std in noise_levels:
        print(f"\n=== Processing noise level: {noise_std:.3f} ===")
        
        # Add noise to the training data
        _, Y_train_noisy = add_noise_to_dataset(X_train, Y_train, noise_std)
        
        # Train the model
        start_time = time.time()
        X_basis, alpha, lengthscales = train_kernel_model(
            X_train, Y_train_noisy, X_train_features, n_basis=n_basis, lam=lam
        )
        train_time = time.time() - start_time
        print(f"Model trained in {train_time:.2f} seconds")
        
        # Convert to JAX arrays for optimization
        X_basis_jax = jnp.array(X_basis)
        alpha_jax = jnp.array(alpha)
        lengthscales_jax = jnp.array(lengthscales)
        
        # Evaluate model on test set
        Y_pred_test = predict_kernel_model(X_test_features, X_basis, alpha, lengthscales)
        test_mse = np.mean((Y_test - Y_pred_test)**2)
        test_nmse = test_mse / np.var(Y_test)
        print(f"Test MSE: {test_mse:.6f}, Normalized MSE: {test_nmse:.6f}")
        
        results["model_metrics"].append({
            "noise_std": noise_std,
            "test_mse": test_mse,
            "test_nmse": test_nmse,
            "train_time": train_time
        })
        
        # Optimize policy for each initial condition
        for scenario, x0 in x0_list:
            print(f"\nOptimizing policy for scenario: {scenario}")
            
            # Find optimal policy
            start_time = time.time()
            policy, model_loss = optimize_policy(
                x0, X_basis_jax, alpha_jax, lengthscales_jax,
                T=policy_T
            )
            opt_time = time.time() - start_time
            
            print(f"Policy optimized in {opt_time:.2f} seconds")
            print(f"Policy: {policy}")
            print(f"Model loss: {model_loss:.6f}")
            
            # Evaluate on true system
            traj_true, true_loss = evaluate_with_true_system(x0, policy, T=eval_T)
            print(f"True system loss: {true_loss:.6f}")
            
            results["policy_metrics"][scenario].append({
                "noise_std": noise_std,
                "policy": policy,
                "model_loss": float(model_loss),
                "optimization_time": opt_time
            })
            
            results["true_system_metrics"][scenario].append({
                "noise_std": noise_std,
                "true_loss": float(true_loss)
            })
        
        # Save intermediate results at each noise level
        np.savez_compressed(
            f"models/noise_impact/results_noise_{noise_std:.3f}.npz",
            noise_std=noise_std,
            X_basis=X_basis,
            alpha=alpha,
            lengthscales=lengthscales,
            test_mse=test_mse,
            test_nmse=test_nmse,
            policies={s: results["policy_metrics"][s][-1]["policy"] for s in results["policy_metrics"]},
            model_losses={s: results["policy_metrics"][s][-1]["model_loss"] for s in results["policy_metrics"]},
            true_losses={s: results["true_system_metrics"][s][-1]["true_loss"] for s in results["policy_metrics"]}
        )
    
    # Save complete results
    np.savez_compressed(
        "models/noise_impact/complete_results.npz",
        noise_levels=noise_levels,
        model_metrics=results["model_metrics"],
        policy_metrics=results["policy_metrics"],
        true_system_metrics=results["true_system_metrics"]
    )
    
    # Create plots showing impact of noise
    # 1. Model MSE vs Noise Level
    plt.figure(figsize=(10, 6))
    mse_values = [m["test_mse"] for m in results["model_metrics"]]
    nmse_values = [m["test_nmse"] for m in results["model_metrics"]]
    
    plt.plot(noise_levels, mse_values, 'o-', label='MSE')
    plt.plot(noise_levels, nmse_values, 's-', label='NMSE')
    plt.xlabel('Noise Standard Deviation')
    plt.ylabel('Error')
    plt.title('Model Error vs Noise Level')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/noise_impact/model_error_vs_noise.png", bbox_inches='tight')
    plt.savefig("figures/noise_impact/model_error_vs_noise.pdf", bbox_inches='tight')
    plt.close()
    
    # 2. Plot true system loss vs noise level for each scenario
    plt.figure(figsize=(10, 6))
    for scenario in results["true_system_metrics"]:
        losses = [m["true_loss"] for m in results["true_system_metrics"][scenario]]
        plt.plot(noise_levels, losses, 'o-', label=f'{scenario}')
    
    plt.xlabel('Noise Standard Deviation')
    plt.ylabel('True System Loss')
    plt.title('Policy Performance Degradation with Noise')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/noise_impact/policy_loss_vs_noise.png", bbox_inches='tight')
    plt.savefig("figures/noise_impact/policy_loss_vs_noise.pdf", bbox_inches='tight')
    plt.close()
    
    # 3. Plot policy parameter changes with noise level
    for scenario in results["policy_metrics"]:
        policies = [m["policy"] for m in results["policy_metrics"][scenario]]
        policies_array = np.array(policies)
        
        plt.figure(figsize=(10, 6))
        for i in range(4):
            param_name = ["p₁ (x)", "p₂ (ẋ)", "p₃ (θ)", "p₄ (θ̇)"][i]
            plt.plot(noise_levels, policies_array[:, i], 'o-', label=param_name)
        
        plt.xlabel('Noise Standard Deviation')
        plt.ylabel('Policy Parameter Value')
        plt.title(f'Policy Parameters vs Noise Level - {scenario}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"figures/noise_impact/policy_params_{scenario}.png", bbox_inches='tight')
        plt.savefig(f"figures/noise_impact/policy_params_{scenario}.pdf", bbox_inches='tight')
        plt.close()
    
    print("\nAnalysis completed and saved to models/noise_impact and figures/noise_impact")

if __name__ == "__main__":
    main()
