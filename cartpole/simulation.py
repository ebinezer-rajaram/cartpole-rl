import numpy as np
from .CartPole import CartPole, remap_angle


def rollout(init_state, T=200, action_fn=None, remap=True, visual=False):
    """
    Simulate the cart-pole system from an initial state over T time steps.

    Parameters:
    - init_state: array-like, shape (4,) -- [x, x_dot, theta, theta_dot]
    - T: int -- number of time steps to simulate
    - action_fn: function f(t, state) -> float -- returns action force at time t
    - remap: bool -- whether to remap theta to [-pi, pi) after each step
    - visual: bool -- whether to render with GUI (uses CartPole's built-in rendering)

    Returns:
    - np.ndarray of shape (T, 4) containing the state trajectory
    """
    env = CartPole(visual=visual)
    env.setState(init_state)

    traj = []
    for t in range(T):
        state = env.getState()
        action = 0.0 if action_fn is None else action_fn(t, state)
        env.performAction(action)

        next_state = env.getState()
        if remap:
            next_state[2] = remap_angle(next_state[2])
        traj.append(next_state.copy())

    return np.array(traj)

def model_rollout(X0, C, T, remap_theta=True):
    """
    Roll out the learned linear model starting from X0.

    Args:
        X0 (ndarray): shape (4,), initial state
        C (ndarray): shape (4, d), learned model matrix (Δ = C X)
        T (int): number of time steps
        remap_theta (bool): whether to remap angle after each step

    Returns:
        ndarray: trajectory of shape (T, 4)
    """
    d = C.shape[1]
    X = np.zeros((T, 4))
    x = X0.copy()

    for t in range(T):
        X[t] = x.copy()
        dx = C @ x[:d]
        x = x + dx
        if remap_theta:
            x[2] = remap_angle(x[2])

    return X

