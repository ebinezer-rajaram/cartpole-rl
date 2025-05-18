import numpy as np
from cartpole.CartPole import CartPole

def collect_dataset(n_samples, low=None, high=None, with_action=False):
    """
    Generate a dataset of (X, Y) or (X, A, Y) transitions.

    Parameters:
    - n_samples: number of data points
    - low, high: bounds for uniform sampling (defaults: 4D cart-pole)
    - with_action: if True, include random force in X

    Returns:
    - X: shape (n_samples, 4) or (n_samples, 5) if with_action
    - Y: shape (n_samples, 4)
    """
    if low is None:
        low = np.array([-5, -10, -np.pi, -15])
    if high is None:
        high = np.array([5, 10, np.pi, 15])

    X_list = []
    Y_list = []

    for _ in range(n_samples):
        x = np.random.uniform(low, high)
        a = np.random.uniform(-20, 20) if with_action else 0.0

        env = CartPole()
        env.setState(x)
        env.performAction(a)
        x_next = env.getState()
        y = x_next - x

        x_full = np.append(x, a) if with_action else x
        X_list.append(x_full)
        Y_list.append(y)

    return np.array(X_list), np.array(Y_list)
