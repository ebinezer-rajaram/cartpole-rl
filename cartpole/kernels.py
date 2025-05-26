import numpy as np

def periodic_kernel(X1, X2, lengthscales, theta_index=2):
    """
    Gaussian kernel with periodic treatment of θ.
    X1: (N, D), X2: (M, D), returns (N, M)
    """
    N, D = X1.shape
    M = X2.shape[0]
    K = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            diff = X1[i] - X2[j]
            # Use sin² periodic distance for theta
            diff[theta_index] = np.sin((X1[i, theta_index] - X2[j, theta_index]) / 2)
            K[i, j] = np.exp(-0.5 * np.sum((diff / lengthscales) ** 2))
    return K

def fit_kernel_model(X, Y, basis_X, lengthscales, lam=1e-4):
    """
    Fit kernel regression model: Δ = K α
    - X: full training inputs, shape (N, D)
    - Y: training targets, shape (N, 4)
    - basis_X: M selected basis inputs, shape (M, D)
    - Returns: α matrix, shape (M, 4)
    """
    K = periodic_kernel(X, basis_X, lengthscales)
    alpha = np.linalg.solve(K.T @ K + lam * np.eye(K.shape[1]), K.T @ Y)
    return alpha

def predict_kernel_model(X_new, basis_X, alpha, lengthscales):
    """
    Predict using trained kernel model.
    - X_new: new inputs, shape (N', D)
    - Returns: predicted Δ, shape (N', 4)
    """
    K_new = periodic_kernel(X_new, basis_X, lengthscales)
    return K_new @ alpha


