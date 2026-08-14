import numpy as np
import pytest
import torch

import grid_policy_sfm_hp100 as GP
import sfm_hp100_features as HF
import sfm_hp100_history as HH


def test_hp100_raster_is_decoupled_from_nominal_polytope_faces():
    center = np.array([0.3, -0.2], np.float32)
    empty = np.zeros((0, 3), np.float32)
    frame16 = HF.hp100_frame(center, empty, n_base=16)
    frame8 = HF.hp100_frame(center, empty, n_base=8)
    assert frame16.shape == (32, 100)
    assert frame16.dtype == np.float32
    assert frame8.shape == frame16.shape
    assert not np.allclose(frame16, frame8)
    assert frame16.min() >= -1.0 and frame16.max() <= 1.0
    with pytest.raises(ValueError, match="at least four"):
        HF.hp100_frame(center, empty, n_base=3)


def test_hp100_raster_supports_only_explicit_predictive_counterfactual():
    center = np.zeros(2, np.float32)
    obstacles = np.array([[1.0, 0.0, 0.2]], np.float32)
    static = HF.hp100_frame(
        center, obstacles, predict_gain=0.0,
        obstacle_velocities=np.zeros((1, 2)), robot_velocity=np.array([1.0, 0.0]),
    )
    predictive = HF.hp100_frame(
        center, obstacles, predict_gain=0.25,
        obstacle_velocities=np.zeros((1, 2)), robot_velocity=np.array([1.0, 0.0]),
    )
    assert predictive.shape == static.shape == (32, 100)
    assert np.any(predictive < static - 1.0e-5)
    assert HF.PREDICT_GAIN == 0.0 and HF.PREDICT_TAU == 1.0
    assert HF.contract()["pedestrian_velocity_in_geometry"] is False
    frame, geometry = HF.hp100_frame(
        center, obstacles, obstacle_velocities=np.zeros((1, 2)),
        robot_velocity=np.array([1.0, 0.0]), return_geometry=True,
    )
    assert np.array_equal(frame, static)
    assert geometry["A"].shape[0] >= 16
    assert geometry["A"].shape[1:] == (2,)
    assert geometry["b"].shape == geometry["margins"].shape
    assert np.allclose(geometry["ref"], center)


def test_current_tangent_hp_is_invariant_to_closing_velocity():
    center = np.array([0.0, 0.0], np.float32)
    obstacles = np.array([[0.8, 0.0, 0.2]], np.float32)
    still = HF.hp100_frame(
        center, obstacles,
        obstacle_velocities=np.zeros((1, 2), np.float32),
        robot_velocity=np.zeros(2, np.float32),
    )
    closing = HF.hp100_frame(
        center, obstacles,
        obstacle_velocities=np.array([[-2.0, 0.0]], np.float32),
        robot_velocity=np.array([2.0, 0.0], np.float32),
    )
    np.testing.assert_array_equal(still, closing)


def test_hp100_raster_is_bitwise_reproducible_from_persisted_narrow_geometry():
    center = np.array([3.434889, 3.678803], np.float32)
    obstacles = np.array([[3.64, 3.68, 0.2]], np.float32)
    frame, geometry = HF.hp100_frame(
        center,
        obstacles,
        obstacle_velocities=np.array([[-2.0, 0.0]], np.float32),
        robot_velocity=np.array([2.0, 0.0], np.float32),
        return_geometry=True,
    )
    theta = -np.pi + (np.arange(32) + 0.5) * 2.0 * np.pi / 32
    radius = (np.arange(100) + 0.5) * HF.R_SENSE / 100
    directions = np.stack((np.cos(theta), np.sin(theta)), axis=1)
    points = center.astype(np.float64)[None, None] + (
        directions[:, None] * radius[None, :, None]
    )
    A = geometry["A"].astype(np.float64)
    b = geometry["b"].astype(np.float64)
    margins = geometry["margins"].astype(np.float64)
    rebuilt = ((b[None] - points.reshape(-1, 2) @ A.T) / margins[None]).min(1)
    rebuilt = np.clip(rebuilt, -1.0, 1.0).reshape(32, 100).astype(np.float32)
    assert np.array_equal(frame, rebuilt)


def test_low5_and_control_history_use_shared_two_unit_limits():
    low = HF.low5([0, 0, 2, -2], [5, 5], 0.5)
    assert np.allclose(low, [1, 1, 1, -1, 0.5])
    history = HF.hist_pad([[2, -2], [4, -4]], K=3)
    assert np.allclose(history, [[0, 0], [1, -1], [1, -1]])
    assert history.min() >= -1.0 and history.max() <= 1.0


def test_hp100_history_is_newest_first_and_episode_leak_free():
    frames = torch.stack([
        torch.full((32, 100), 10.0),
        torch.full((32, 100), 20.0),
        torch.full((32, 100), 11.0),
        torch.full((32, 100), 21.0),
    ])
    episodes = torch.tensor([1, 2, 1, 2])
    steps = torch.tensor([0, 0, 1, 1])
    built = HH.build_hp100(frames, episodes, steps)
    assert built.shape == (4, 10, 32, 100)
    assert torch.all(built[2, 0] == 11)
    assert torch.all(built[2, 1:] == 10)
    assert torch.all(built[3, 0] == 21)
    assert torch.all(built[3, 1:] == 20)

    online = HH.Hp100History()
    assert torch.all(online.append(frames[0]) == 10)
    second = online.append(frames[2])
    assert torch.all(second[0] == 11)
    assert torch.all(second[1:] == 10)
    online.reset()
    assert torch.all(online.append(frames[1]) == 20)


def test_hp100_history_rejects_interior_gaps():
    frames = torch.zeros(2, 32, 100)
    with pytest.raises(ValueError, match="missing past step 1"):
        HH.build_hp100(frames, [3, 3], [0, 2])


def test_hp100_policy_shapes_and_no_radial_pooling():
    policy = GP.build_sfm_hp100_policy()
    batch = 3
    grid = torch.randn(batch, 10, 32, 100)
    low = torch.randn(batch, 5)
    hist = torch.randn(batch, 16, 2)
    conv = policy.grid_conv(grid)
    pooled = policy.angular_pool(conv)
    assert conv.shape == (batch, 32, 32, 100)
    assert pooled.shape == (batch, 32, 4, 100)
    context = policy.ctx_from(grid, low, hist)
    assert context.shape == (batch, 176)
    output = policy(torch.randn(batch, 20), torch.rand(batch), context)
    assert output.shape == (batch, 20)
    assert policy.head.out_features == 20


def test_hp100_policy_strict_config_params_and_round_trip(tmp_path):
    policy = GP.build_sfm_hp100_policy()
    config = policy.config()
    assert config["arch"] == "v3-sfm-hp100-residual"
    assert config["grid_shape"] == (10, 32, 100)
    assert config["conv_channels"] == (10, 16, 32)
    assert config["radial_pool"] is None
    assert config["visual_token"] == 128
    assert config["gru_dim"] == 16 and config["low_token"] == 48
    assert config["width"] == 256 and config["output_dim"] == 20
    assert config["u_max"] == 2.0 and config["v_max"] == 2.0
    assert config["predict_gain"] == 0.0
    assert sum(parameter.numel() for parameter in policy.parameters()) == 1_978_068

    path = tmp_path / "hp100.pt"
    GP.save_sfm_hp100_policy(policy, path, extra={"epoch": 7})
    restored, checkpoint = GP.load_sfm_hp100_policy(path)
    assert checkpoint["epoch"] == 7
    assert restored.config() == config
    for name, value in policy.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])

    payload = torch.load(path, weights_only=False)
    payload["config"]["radial_pool"] = 4
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match=r"differing=\['radial_pool'\]"):
        GP.load_sfm_hp100_policy(bad)
    with pytest.raises(ValueError, match="cannot replace"):
        GP.save_sfm_hp100_policy(policy, tmp_path / "overwrite.pt", extra={"config": {}})


def test_hp100_head_only_expansion_freezes_every_pre_head_parameter():
    policy = GP.build_sfm_hp100_policy()
    GP.configure_head_only_expansion(policy)
    trainable = {name for name, value in policy.named_parameters() if value.requires_grad}
    assert trainable == {"head.weight", "head.bias"}
    assert all(not value.requires_grad for value in policy.grid_conv.parameters())
    assert all(not value.requires_grad for value in policy.grid_projection.parameters())
    assert all(not value.requires_grad for value in policy.gru.parameters())
    assert all(not value.requires_grad for value in policy.enc_low.parameters())
    assert all(not value.requires_grad for value in policy.trunk.parameters())
