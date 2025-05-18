import pytest
import numpy as np
from cartpole.scanning import perform_single_step, scan_1d, scan_2d
from cartpole.CartPole import CartPole


@pytest.fixture
def test_state():
    # Default hanging-down state
    return np.array([0.0, 0.0, np.pi, 0.0])  


@pytest.fixture
def scan_ranges():
    return {
        0: (-5, 5),     # x
        1: (-10, 10),   # x_dot
        2: (0, 2*np.pi),# theta
        3: (-15, 15)    # theta_dot
    }


def test_perform_single_step_full_state(test_state):
    # Test that perform_single_step returns the full state correctly
    result = perform_single_step(test_state, return_delta=False)
    
    # Check result shape and type
    assert len(result) == 4
    assert isinstance(result, np.ndarray)
    
    # Check that result is different from original state (simulation progressed)
    assert not np.array_equal(result, test_state)
    

def test_perform_single_step_delta(test_state):
    # Test that perform_single_step returns the state delta correctly
    result = perform_single_step(test_state, return_delta=True)
    
    # Check result shape and type
    assert len(result) == 4
    assert isinstance(result, np.ndarray)
    
    # Get the raw next state
    env = CartPole()
    env.setState(test_state)
    env.performAction(0.0)
    next_state = env.getState()
    
    # Check that delta is correctly computed
    expected_delta = next_state - test_state
    np.testing.assert_allclose(result, expected_delta, rtol=1e-10)


def test_scan_1d(test_state, scan_ranges):
    # Test 1D scanning for each state variable
    for index in range(4):
        # Create scan values
        scan_values = np.linspace(
            scan_ranges[index][0], 
            scan_ranges[index][1], 
            10
        )
        
        # Test with full state
        x_vals, y_vals = scan_1d(index, scan_values, test_state, return_delta=False)
        
        # Check shapes
        assert len(x_vals) == len(scan_values)
        assert y_vals.shape == (len(scan_values), 4)
        
        # Check x_vals are correctly set
        np.testing.assert_allclose(x_vals, scan_values)
        
        # Test with delta
        x_vals, y_deltas = scan_1d(index, scan_values, test_state, return_delta=True)
        assert y_deltas.shape == (len(scan_values), 4)
        
        # Verify difference between full state and delta
        for i, val in enumerate(scan_values):
            state = test_state.copy()
            state[index] = val
            full_state_result = perform_single_step(state, return_delta=False)
            delta_result = perform_single_step(state, return_delta=True)
            
            # Verify that the delta is the difference between next state and current
            np.testing.assert_allclose(delta_result, full_state_result - state, rtol=1e-10)
            
            # Verify that the scan results match direct function calls
            np.testing.assert_allclose(y_vals[i], full_state_result, rtol=1e-10)
            np.testing.assert_allclose(y_deltas[i], delta_result, rtol=1e-10)


def test_scan_2d(test_state, scan_ranges):
    # Test 2D scanning for a pair of state variables
    i, j = 0, 1  # x and x_dot
    range_i = scan_ranges[i]
    range_j = scan_ranges[j]
    grid_resolution = 5  # Small for testing
    
    # Test with full state
    x_coords, y_vals = scan_2d(i, j, range_i, range_j, test_state, 
                              return_delta=False, grid_resolution=grid_resolution)
    
    # Check shapes
    assert x_coords.shape == (grid_resolution**2, 2)
    assert y_vals.shape == (grid_resolution**2, 4)
    
    # Test with delta
    x_coords, y_deltas = scan_2d(i, j, range_i, range_j, test_state, 
                                return_delta=True, grid_resolution=grid_resolution)
    
    # Check shapes again
    assert y_deltas.shape == (grid_resolution**2, 4)
    
    # Verify a few points to make sure deltas are computed correctly
    for idx in [0, grid_resolution**2 // 2, grid_resolution**2 - 1]:
        state = test_state.copy()
        state[i] = x_coords[idx][0]
        state[j] = x_coords[idx][1]
        
        full_state_result = perform_single_step(state, return_delta=False)
        delta_result = perform_single_step(state, return_delta=True)
        
        # Verify full state results
        np.testing.assert_allclose(y_vals[idx], full_state_result, rtol=1e-10)
        
        # Verify delta results
        np.testing.assert_allclose(y_deltas[idx], delta_result, rtol=1e-10)
