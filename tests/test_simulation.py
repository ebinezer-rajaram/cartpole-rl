import numpy as np
import pytest
from cartpole.simulation import rollout

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
