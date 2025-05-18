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
