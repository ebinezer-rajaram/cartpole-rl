import numpy as np
from cartpole.CartPole import CartPole

def perform_single_step(state, return_delta=False):
    env = CartPole()
    env.setState(state)
    env.performAction(0.0)
    new_state = env.getState()
    return new_state - state if return_delta else new_state

def scan_1d(index, scan_vals, base_state, return_delta=False):
    X_vals = []
    Y_vals = []
    for val in scan_vals:
        state = base_state.copy()
        state[index] = val
        result = perform_single_step(state, return_delta)
        X_vals.append(val)
        Y_vals.append(result)
    return np.array(X_vals), np.array(Y_vals)

def scan_2d(i, j, range_i, range_j, base_state, return_delta=False, grid_resolution=30):
    scan_i = np.linspace(range_i[0], range_i[1], grid_resolution)
    scan_j = np.linspace(range_j[0], range_j[1], grid_resolution)
    grid_i, grid_j = np.meshgrid(scan_i, scan_j)
    points = np.vstack([grid_i.ravel(), grid_j.ravel()]).T

    X_coords = []
    Y_vals = []

    for p in points:
        state = base_state.copy()
        state[i] = p[0]
        state[j] = p[1]
        result = perform_single_step(state, return_delta)
        X_coords.append(p)
        Y_vals.append(result)

    return np.array(X_coords), np.array(Y_vals)


def scan_model_vs_true(index, scan_vals, base_state, model_predict_fn):
    """
    Scan over one input dimension, compare model and true ΔY.
    
    Args:
        index: int — index of the input to vary
        scan_vals: array — values to scan over
        base_state: 1D array of shape (4,) — fixed state
        model_predict_fn: callable(state: (4,)) → (4,) — predicts ΔY
    
    Returns:
        scan_vals: array of input values
        Y_true: array (len(scan_vals), 4)
        Y_pred: array (len(scan_vals), 4)
    """
    Y_true = []
    Y_pred = []

    for val in scan_vals:
        state = base_state.copy()
        state[index] = val

        # True ΔY from simulator
        env = CartPole()
        env.setState(state)
        env.performAction(0.0)
        delta_true = env.getState() - state

        # Model prediction
        delta_pred = model_predict_fn(state)

        Y_true.append(delta_true)
        Y_pred.append(delta_pred)

    return np.array(scan_vals), np.array(Y_true), np.array(Y_pred)

def scan2d_model_vs_true(i, j, range_i, range_j, base_state, model_predict_fn, grid_resolution=30):
    """
    Generate 2D grid scan over input dims i and j.

    Returns:
        X_coords: (N, 2) — input grid values
        Y_true: (N, 4) — true Δ
        Y_pred: (N, 4) — model-predicted Δ
    """
    scan_i = np.linspace(range_i[0], range_i[1], grid_resolution)
    scan_j = np.linspace(range_j[0], range_j[1], grid_resolution)
    grid_i, grid_j = np.meshgrid(scan_i, scan_j)
    points = np.vstack([grid_i.ravel(), grid_j.ravel()]).T

    X_coords = []
    Y_true = []
    Y_pred = []

    for p in points:
        state = base_state.copy()
        state[i] = p[0]
        state[j] = p[1]

        env = CartPole()
        env.setState(state)
        env.performAction(0.0)
        dy_true = env.getState() - state
        dy_pred = model_predict_fn(state)

        X_coords.append(p)
        Y_true.append(dy_true)
        Y_pred.append(dy_pred)

    return np.array(X_coords), np.array(Y_true), np.array(Y_pred)

def scan_model_vs_true_action(index, scan_vals, base_state_action, model_predict_fn):
    Y_true = []
    Y_pred = []

    for val in scan_vals:
        state_action = base_state_action.copy()
        state_action[index] = val

        state = state_action[:4]
        action = state_action[4]

        env = CartPole()
        env.setState(state)
        env.performAction(action)
        delta_true = env.getState() - state
        delta_pred = model_predict_fn(state_action)

        Y_true.append(delta_true)
        Y_pred.append(delta_pred)

    return np.array(scan_vals), np.array(Y_true), np.array(Y_pred)


def scan2d_model_vs_true_action(i, j, range_i, range_j, base_state_action, model_predict_fn, grid_resolution=30):
    scan_i = np.linspace(range_i[0], range_i[1], grid_resolution)
    scan_j = np.linspace(range_j[0], range_j[1], grid_resolution)
    grid_i, grid_j = np.meshgrid(scan_i, scan_j)
    points = np.vstack([grid_i.ravel(), grid_j.ravel()]).T

    X_coords = []
    Y_true = []
    Y_pred = []

    for p in points:
        state_action = base_state_action.copy()
        state_action[i] = p[0]
        state_action[j] = p[1]

        state = state_action[:4]
        action = state_action[4]

        env = CartPole()
        env.setState(state)
        env.performAction(action)
        delta_true = env.getState() - state
        delta_pred = model_predict_fn(state_action)

        X_coords.append(p)
        Y_true.append(delta_true)
        Y_pred.append(delta_pred)

    return np.array(X_coords), np.array(Y_true), np.array(Y_pred)



