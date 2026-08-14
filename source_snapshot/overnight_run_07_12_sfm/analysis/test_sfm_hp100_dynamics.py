import numpy as np
import pytest
import torch

import sfm_hp100_dynamics as D


def test_numpy_caps_action_and_physical_velocity_componentwise():
    state = np.array([1.0, -1.0, 1.95, -2.4], np.float32)
    action = np.array([9.0, -9.0], np.float32)
    result = D.step_numpy(state, action)
    np.testing.assert_allclose(result[:2], [1.205, -1.21], atol=1.0e-6)
    np.testing.assert_allclose(result[2:], [2.0, -2.0], atol=1.0e-6)


def test_numpy_and_torch_batched_steps_match():
    state = np.array([
        [0.0, 0.0, 0.2, -0.3],
        [1.0, 2.0, 2.2, -1.9],
    ], np.float32)
    action = np.array([[0.4, -0.8], [-8.0, 8.0]], np.float32)
    expected = D.step_numpy(state, action)
    actual = D.step_torch(torch.from_numpy(state), torch.from_numpy(action)).numpy()
    np.testing.assert_allclose(actual, expected, atol=1.0e-7)
    assert np.max(np.abs(actual[:, 2:])) <= D.V_MAX


def test_torch_step_preserves_gradient():
    action = torch.tensor([[0.1, -0.2]], requires_grad=True)
    result = D.step_torch(torch.zeros(1, 4), action)
    result.sum().backward()
    assert action.grad is not None
    assert torch.isfinite(action.grad).all()


@pytest.mark.parametrize("helper,value", [
    (D.clip_action_numpy, np.zeros(3)),
    (D.clip_velocity_numpy, np.zeros(3)),
    (D.clip_action_torch, torch.zeros(3)),
    (D.clip_velocity_torch, torch.zeros(3)),
])
def test_clip_helpers_reject_wrong_shape(helper, value):
    with pytest.raises(ValueError, match="last dimension 2"):
        helper(value)
