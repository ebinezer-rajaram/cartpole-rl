import pytest
import numpy as np
from cartpole import CartPole, remap_angle, loss

@pytest.fixture
def cartpole_env():
    return CartPole()

def test_initial_state_values(cartpole_env):
    cartpole_env.reset()
    state = cartpole_env.getState()
    assert np.allclose(state, [0.0, 0.0, np.pi, 0.0])

def test_set_and_get_state(cartpole_env):
    state = [1.0, 2.0, 0.5, -0.5]
    cartpole_env.setState(state)
    np.testing.assert_allclose(cartpole_env.getState(), state)

def test_perform_action_changes_state(cartpole_env):
    cartpole_env.reset()
    state_before = cartpole_env.getState().copy()
    cartpole_env.performAction(1.0)
    state_after = cartpole_env.getState()
    assert not np.allclose(state_before, state_after)

def test_remap_angle_function():
    assert np.isclose(remap_angle(np.pi + 2 * np.pi), np.pi)
    assert np.isclose(remap_angle(-np.pi - 2 * np.pi), -np.pi)
    assert np.isclose(remap_angle(0), 0)

def test_loss_function_zero_state():
    state = np.zeros(4)
    assert np.isclose(loss(state), 0)

def test_loss_function_nonzero_state():
    state = np.array([1.0, 0.0, 0.0, 0.0])
    assert loss(state) > 0

def test_loss_member_function(cartpole_env):
    cartpole_env.reset()
    assert np.isclose(cartpole_env.loss(), loss(cartpole_env.getState()))

def test_visual_flag_initialization():
    env = CartPole(visual=True)
    assert env.visual is True

def test_perform_action_with_large_action(cartpole_env):
    cartpole_env.reset()
    state_before = cartpole_env.getState().copy()
    cartpole_env.performAction(1e6)
    state_after = cartpole_env.getState()
    assert not np.allclose(state_before, state_after)

def test_remap_angle_member(cartpole_env):
    cartpole_env.pole_angle = 3 * np.pi
    cartpole_env.remap_angle()
    assert -np.pi <= cartpole_env.pole_angle <= np.pi
