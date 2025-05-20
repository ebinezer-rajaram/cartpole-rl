import numpy as np
import pytest
from cartpole.simulation import rollout, model_rollout

def test_rollout_returns_correct_shape():
    init_state = np.array([0.0, 0.0, np.pi, 0.0])
    T = 10
    traj = rollout(init_state, T=T)
    assert isinstance(traj, np.ndarray)
    assert traj.shape == (T, 4)

def test_rollout_state_changes():
    init_state = np.array([0.0, 0.0, np.pi, 0.01])
    traj = rollout(init_state, T=5)
    # At least one state should change from the initial state
    assert not np.allclose(traj, np.tile(init_state, (5, 1)))

def test_rollout_with_action_fn():
    def action_fn(t, state):
        return 1.0 if t % 2 == 0 else -1.0
    init_state = np.array([0.0, 0.0, np.pi, 0.0])
    traj = rollout(init_state, T=10, action_fn=action_fn)
    assert traj.shape == (10, 4)
    # Should not be all the same state
    assert not np.allclose(traj, np.tile(init_state, (10, 1)))

def test_rollout_remap_angle():
    init_state = np.array([0.0, 0.0, 3 * np.pi, 0.0])
    traj = rollout(init_state, T=1, remap=True)
    # Angle should be remapped to [-pi, pi]
    assert -np.pi <= traj[0, 2] <= np.pi

def test_model_rollout_returns_correct_shape():
    init_state = np.array([0.0, 0.0, np.pi, 0.0])
    # Simple C matrix that keeps state mostly unchanged
    C = np.array([
        [0.01, 0.0, 0.0, 0.0],  # slight change to x
        [0.0, 0.0, 0.0, 0.0],   # no change to x_dot
        [0.0, 0.0, 0.01, 0.0],  # slight change to theta
        [0.0, 0.0, 0.0, 0.0]    # no change to theta_dot
    ])
    T = 10
    traj = model_rollout(init_state, C, T)
    
    assert isinstance(traj, np.ndarray)
    assert traj.shape == (T, 4)

def test_model_rollout_state_evolves_correctly():
    init_state = np.array([1.0, 0.0, 0.0, 0.0])
    # Define a simple model: x increases by 0.1 each step
    C = np.zeros((4, 4))
    C[0, 0] = 0.1  # x increases by 0.1*x each step
    
    T = 5
    traj = model_rollout(init_state, C, T)
    
    # Check that x increases exponentially (1, 1.1, 1.21, 1.331, 1.4641)
    expected_x_values = np.array([1.0, 1.1, 1.21, 1.331, 1.4641])
    np.testing.assert_allclose(traj[:, 0], expected_x_values, rtol=1e-5)
    
    # Other state variables should remain at 0
    assert np.allclose(traj[:, 1:], 0)

def test_model_rollout_angle_remapping():
    init_state = np.array([0.0, 0.0, np.pi - 0.1, 0.0])
    # Model where theta increases and eventually crosses pi
    C = np.zeros((4, 4))
    C[2, 2] = 0.2  # theta increases by 0.2*theta each step
    
    T = 10
    
    # With remapping
    traj_remapped = model_rollout(init_state, C, T, remap_theta=True)
    # Angles should always be in [-pi, pi]
    assert np.all(traj_remapped[:, 2] >= -np.pi)
    assert np.all(traj_remapped[:, 2] < np.pi)
    
    # Without remapping
    traj_not_remapped = model_rollout(init_state, C, T, remap_theta=False)
    # Angles should exceed pi
    assert np.any(traj_not_remapped[:, 2] > np.pi)

def test_model_rollout_with_reduced_dimension():
    init_state = np.array([1.0, 2.0, 3.0, 4.0])
    # Model that only uses first 2 dimensions
    C = np.zeros((4, 2))
    C[0, 0] = 0.1  # x affected by x
    C[1, 1] = 0.1  # x_dot affected by x_dot
    
    T = 5
    traj = model_rollout(init_state, C, T)
    
    assert traj.shape == (T, 4)
    # First dimension should change
    assert not np.allclose(traj[:, 0], init_state[0])
    # Second dimension should change
    assert not np.allclose(traj[:, 1], init_state[1])

def test_model_rollout_coupled_dimensions():
    init_state = np.array([1.0, 0.0, 0.1, 0.0])
    # Example physics-inspired model where:
    # - x_dot is affected by x (velocity depends on position)
    # - theta_dot is affected by theta (angular velocity depends on angle)
    C = np.zeros((4, 4))
    C[1, 0] = 0.1  # x affects x_dot
    C[3, 2] = 0.2  # theta affects theta_dot
    
    T = 5
    traj = model_rollout(init_state, C, T)
    
    # Check x_dot and theta_dot change due to coupling
    assert not np.allclose(traj[:, 1], init_state[1])
    assert not np.allclose(traj[:, 3], init_state[3])
    
    # Verify specific dynamics pattern (acceleration increases with time)
    assert abs(traj[2, 1]) > abs(traj[1, 1])  # Increasing rate of change

