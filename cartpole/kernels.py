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


