import inspect
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

import sfm_hp100_kazuki as K
import sfm_hp100_kazuki_eval as EVAL
import sfm_kazuki as LEGACY


class _Policy:
    H_pred = 10
    d = 20
    u_max = 2.0

    def _expand_ctx(self, context, count):
        return context.reshape(1, -1).expand(count, -1)

    def forward(self, value, time, context):
        del time
        return -0.1 * value + 0.01 * context[:, :1]


def test_legacy_rollout_defaults_and_numerics_remain_unbounded():
    assert (
        inspect.signature(LEGACY.guided_generate).parameters["rollout_fn"].default
        is LEGACY.di_rollout_t
    )
    assert (
        inspect.signature(LEGACY.flow_mppi_refine).parameters["rollout_fn"].default
        is LEGACY.di_rollout_t
    )
    state = np.array([0.0, 0.0, 1.95, 0.0], np.float32)
    controls = torch.tensor([[[9.0, 0.0], [9.0, 0.0]]])
    legacy_positions, legacy_velocity = LEGACY.di_rollout_t(state, controls)
    clipped_positions, clipped_velocity = K.clipped_di_rollout_t(state, controls)
    assert float(legacy_velocity[0, -1, 0]) == pytest.approx(3.75)
    assert float(clipped_velocity[0, -1, 0]) == pytest.approx(2.0)
    assert float(legacy_positions[0, -1, 0]) > float(clipped_positions[0, -1, 0])


def test_explicit_legacy_rollout_hook_is_numerically_identical_to_default():
    policy = _Policy()
    context = torch.zeros(3)
    state = np.zeros(4, np.float32)
    goal = torch.tensor([6.0, 6.0])
    pedestrian_prediction = torch.full((1, 10, 2), 20.0)
    pedestrian_velocity = torch.zeros(1, 2)
    initial = torch.randn(4, 20, generator=torch.Generator().manual_seed(19))
    config = replace(K.locked_config(), n_sample=4, n_elite=2, n_copy=3)
    positional = (
        policy, context, state, goal, pedestrian_prediction,
        pedestrian_velocity, 0.25, initial, config.ode_times, config,
    )
    default, _, _, _ = LEGACY.guided_generate(*positional)
    explicit, _, _, _ = LEGACY.guided_generate(
        *positional, rollout_fn=LEGACY.di_rollout_t,
    )
    torch.testing.assert_close(default, explicit, rtol=0, atol=0)

    generated = torch.clamp(default.reshape(4, 10, 2) * 2.0, -2.0, 2.0)
    torch.manual_seed(23)
    default_refined, _ = LEGACY.flow_mppi_refine(
        policy, state, goal, torch.zeros(1, 2), pedestrian_velocity,
        pedestrian_prediction, 0.25, generated, None, config,
    )
    torch.manual_seed(23)
    explicit_refined, _ = LEGACY.flow_mppi_refine(
        policy, state, goal, torch.zeros(1, 2), pedestrian_velocity,
        pedestrian_prediction, 0.25, generated, None, config,
        rollout_fn=LEGACY.di_rollout_t,
    )
    torch.testing.assert_close(default_refined, explicit_refined, rtol=0, atol=0)


def test_locked_comparator_has_no_shield_or_gamma_retuning():
    config = K.locked_config()
    assert tuple(dict(K.LOCKED_CONFIG_ITEMS)) == tuple(
        LEGACY.KazukiConfig.__dataclass_fields__
    )
    assert config.to_dict() == dict(K.LOCKED_CONFIG_ITEMS)
    assert config.safe_coefs == (0.3,)
    assert config.goal_coef == 0.5
    assert not config.output_filter
    assert not config.exact_sfm_step_filter
    assert not config.hard_clearance_select
    assert config.safe_coef_gamma_span == 0.0
    assert config.goal_coef_gamma_span == 0.0
    assert config.controller_gammas == ()


def test_locked_comparator_fails_closed_on_config_schema_drift(monkeypatch):
    drifted = dict(LEGACY.KazukiConfig.__dataclass_fields__)
    drifted["new_unreviewed_default"] = object()
    monkeypatch.setattr(LEGACY.KazukiConfig, "__dataclass_fields__", drifted)
    with pytest.raises(RuntimeError, match="schema changed"):
        K.locked_config()


def test_kazuki_eval_preflight_authenticates_checkpoint_and_refuses_overwrite(
    tmp_path, monkeypatch,
):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"fixed checkpoint")
    checkpoint_sha = EVAL.RAW.sha256_file(checkpoint)
    git = dict(root="/frozen", head="a" * 40, clean=True, status="")
    monkeypatch.setattr(EVAL, "_git_provenance", lambda: git)
    monkeypatch.setattr(EVAL, "_source_hashes", lambda: {"evaluator": {"sha256": "b" * 64}})

    inputs = EVAL._preflight(
        checkpoint, checkpoint_sha, tmp_path / "result.json", "a" * 40,
    )
    assert inputs["checkpoint_sha256"] == checkpoint_sha
    assert inputs["git"] == git
    with pytest.raises(RuntimeError, match="checkpoint SHA-256 mismatch"):
        EVAL._preflight(checkpoint, "0" * 64, tmp_path / "other.json", "a" * 40)

    (tmp_path / "result.json").write_text("old")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        EVAL._preflight(
            checkpoint, checkpoint_sha, tmp_path / "result.json", "a" * 40,
        )


def test_kazuki_eval_records_pins_and_second_run_refuses_output(tmp_path, monkeypatch):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"fixed checkpoint")
    checkpoint_sha = EVAL.RAW.sha256_file(checkpoint)
    git = dict(root="/frozen", head="c" * 40, clean=True, status="")
    sources = {"evaluator": {"path": "/frozen/eval.py", "sha256": "d" * 64}}
    monkeypatch.setattr(EVAL, "_git_provenance", lambda: git)
    monkeypatch.setattr(EVAL, "_source_hashes", lambda: sources)
    monkeypatch.setattr(
        EVAL, "evaluate",
        lambda *args, **kwargs: ([], {"pooled": {"n": 0}, "per_gamma": {}}),
    )
    output = tmp_path / "result.json"
    argv = [
        "--checkpoint", str(checkpoint),
        "--expected-checkpoint-sha256", checkpoint_sha,
        "--expected-source-commit", "c" * 40,
        "--scene-profile", "matched_id", "--ep0", "10", "--M", "1",
        "--device", "cpu", "--out", str(output),
    ]
    EVAL.main(argv)
    payload = json.loads(output.read_text())
    assert payload["checkpoint_sha256"] == checkpoint_sha
    assert payload["source"] == {"git": git, "files": sources}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        EVAL.main(argv)


def test_deploy_uses_hp100_observation_and_caps_actual_execution(monkeypatch):
    calls = {}

    class Policy:
        H_pred = 10
        d = 20
        u_max = 2.0

        def ctx_from(self, hp, low, history):
            calls["context_shapes"] = (
                tuple(hp.shape), tuple(low.shape), tuple(history.shape),
            )
            return torch.zeros(1, 176)

    monkeypatch.setattr(K.SS, "make_humans", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(
        K.SS, "collect_humans",
        lambda humans: (
            np.array([[100.0, 100.0]], np.float32),
            np.array([[0.25, -0.25]], np.float32),
        ),
    )
    monkeypatch.setattr(K.SS, "advance_humans", lambda humans, state: None)

    def fake_hp(robot_xy, obstacles, **kwargs):
        calls["hp_kwargs"] = kwargs
        return np.zeros((32, 100), np.float32)

    monkeypatch.setattr(K.HPF, "hp100_frame", fake_hp)

    def fake_guided(*args, **kwargs):
        calls["guided_rollout"] = kwargs["rollout_fn"]
        latent = args[7]
        return torch.zeros_like(latent), [], None, None

    def fake_refine(*args, **kwargs):
        calls["refine_rollout"] = kwargs["rollout_fn"]
        return torch.full((10, 2), 9.0), None

    monkeypatch.setattr(K.BASE, "guided_generate", fake_guided)
    monkeypatch.setattr(K.BASE, "flow_mppi_refine", fake_refine)

    result = K.kazuki_hp100_deploy(
        Policy(), episode=123, gamma=0.5, scene_profile="matched_id",
        T=1, device="cpu",
    )
    assert calls["context_shapes"] == ((1, 10, 32, 100), (1, 5), (1, 16, 2))
    assert calls["hp_kwargs"]["n_base"] == 16
    assert calls["hp_kwargs"]["predict_gain"] == 0.0
    np.testing.assert_allclose(calls["hp_kwargs"]["robot_velocity"], [0.0, 0.0])
    np.testing.assert_allclose(
        calls["hp_kwargs"]["obstacle_velocities"], [[0.25, -0.25]],
    )
    assert calls["guided_rollout"] is K.clipped_di_rollout_t
    assert calls["refine_rollout"] is K.clipped_di_rollout_t
    np.testing.assert_allclose(result["controls"], [[2.0, 2.0]])
    np.testing.assert_allclose(result["states"][-1], [0.01, 0.01, 0.2, 0.2])
