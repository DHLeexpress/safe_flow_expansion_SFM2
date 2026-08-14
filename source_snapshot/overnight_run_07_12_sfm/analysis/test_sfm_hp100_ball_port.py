from __future__ import annotations

from argparse import Namespace
import json

import numpy as np
import pytest
import torch

import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_ball_launch as LAUNCH
from sfm_hp100_ball_core.provenance import (
    PATCHED_PORT_SHA256,
    SOURCE_SHA256,
    assert_vendored_core,
)
import sfm_hp100_dynamics as DYN
import sfm_metrics2 as VERIFY
import sfm_scene as SS


def _args(**changes):
    values = dict(
        rounds=10, parallel_episodes=16, verifier_workers=8,
        max_retry_batches=32, batch_size=64,
        optimizer_steps_per_round=None, microbatch_repeats=10,
        learning_rate=1.0e-6, gp_buffer_cap=1344, initial_beta=0.01,
        ess_target=0.1, flow_base_std=1.0,
        negative_alpha=0.001, paired_noised_representation=True,
        execution_step_margin_weight=70_000.0,
        successful_trajectories_per_gamma=1, seed=2,
    )
    values.update(changes)
    return Namespace(**values)


def _packed(task, robot, ped_xy, ped_vel):
    token = np.zeros(GPS.LOW_TOKEN + GPS.VISUAL_TOKEN, np.float32)
    return torch.from_numpy(np.concatenate([
        token, np.asarray(robot, np.float32),
        np.asarray(ped_xy, np.float32).reshape(-1),
        np.asarray(ped_vel, np.float32).reshape(-1),
    ]))


def test_vendored_ball_core_is_byte_authenticated():
    provenance = assert_vendored_core()
    assert provenance["sha256"] == PATCHED_PORT_SHA256
    assert provenance["base_source_sha256"] == SOURCE_SHA256
    assert provenance["source_worktree_was_dirty"] is True
    assert provenance["semantic_delta_manifest_sha256"]


def test_head_only_means_only_final_linear_head():
    policy = GPS.GridSFMHP100FlowPolicy()
    GPS.configure_head_only_expansion(policy)
    adapter = PORT.HP100ExpansionPolicy(policy)
    trainable = PORT.assert_head_only(adapter)
    assert trainable == ["policy.head.bias", "policy.head.weight"]
    assert all(not parameter.requires_grad for parameter in policy.trunk.parameters())
    assert all(not parameter.requires_grad for parameter in policy.grid_conv.parameters())
    assert all(not parameter.requires_grad for parameter in policy.gru.parameters())


def test_load_head_only_configures_and_asserts(monkeypatch):
    policy = GPS.GridSFMHP100FlowPolicy()
    monkeypatch.setattr(
        GPS, "load_sfm_hp100_policy", lambda checkpoint, device: (policy, {"ok": True}),
    )
    adapter, _, payload = PORT.load_head_only("unused.pt", device="cpu")
    assert payload == {"ok": True}
    assert PORT.assert_head_only(adapter) == ["policy.head.bias", "policy.head.weight"]


def test_last_block_and_head_scope_is_exact_and_freezes_everything_else():
    policy = GPS.GridSFMHP100FlowPolicy()
    adapter = PORT.HP100ExpansionPolicy(policy)
    parameters = adapter.expansion_optimizer_parameters("last_block_and_head")
    trainable = {
        name for name, parameter in adapter.named_parameters()
        if parameter.requires_grad
    }
    expected = {
        *(f"policy.trunk.blocks.1.{name}"
          for name, _ in policy.trunk.blocks[1].named_parameters()),
        *(f"policy.head.{name}" for name, _ in policy.head.named_parameters()),
    }
    assert trainable == expected
    assert {id(parameter) for parameter in parameters} == {
        id(parameter) for parameter in adapter.parameters()
        if parameter.requires_grad
    }
    assert sum(parameter.numel() for parameter in parameters) == 137_236
    assert all(
        not parameter.requires_grad
        for parameter in policy.trunk.blocks[0].parameters()
    )


def test_trunk_and_head_scope_trains_all_flow_layers_and_freezes_conditions():
    policy = GPS.GridSFMHP100FlowPolicy()
    adapter = PORT.HP100ExpansionPolicy(policy)
    parameters = adapter.expansion_optimizer_parameters("trunk_and_head")
    trainable = {
        name for name, parameter in adapter.named_parameters()
        if parameter.requires_grad
    }
    expected = {
        *(f"policy.trunk.inp.{name}"
          for name, _ in policy.trunk.inp.named_parameters()),
        *(f"policy.trunk.blocks.{block_index}.{name}"
          for block_index, block in enumerate(policy.trunk.blocks)
          for name, _ in block.named_parameters()),
        *(f"policy.head.{name}" for name, _ in policy.head.named_parameters()),
    }
    assert trainable == expected
    assert {id(parameter) for parameter in parameters} == {
        id(parameter) for parameter in adapter.parameters()
        if parameter.requires_grad
    }
    assert all(not parameter.requires_grad for parameter in policy.gru.parameters())
    assert all(not parameter.requires_grad for parameter in policy.enc_low.parameters())
    assert all(
        not parameter.requires_grad for parameter in policy.grid_conv.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in policy.grid_projection.parameters()
    )


def test_rbf_calibration_uses_50_gamma_and_trajectory_balanced_training_contexts(
    monkeypatch,
):
    class FakeDataset:
        def __init__(self):
            self.gamma_rows = []
            self.episodes = []
            self.source_rows = []
            self.items = []
            for gamma_index in range(len(SS.GAMMAS)):
                for lineage in range(10):
                    for step in range(2):
                        self.gamma_rows.append(gamma_index)
                        self.episodes.append(1000 * gamma_index + lineage)
                        self.source_rows.append(100 * lineage + step)
                        low5 = torch.tensor([0, 0, 0, 0, SS.GAMMAS[gamma_index]])
                        self.items.append((
                            torch.full((10, 32, 100), float(lineage)), low5,
                            torch.zeros((16, 2)), torch.zeros((10, 2)),
                            torch.tensor(1000 * gamma_index + lineage),
                            torch.tensor(gamma_index),
                        ))
            self.gamma_rows = torch.tensor(self.gamma_rows)
            self.episodes = torch.tensor(self.episodes)
            self.source_rows = torch.tensor(self.source_rows)

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    class FakePolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def ctx_from(self, hp100, low5, hist):
            del hist
            return torch.stack((hp100.mean(), low5[-1])).reshape(1, 2)

    class FakeAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = FakePolicy()

        def sample(self, context, count, generator, base_std):
            del generator
            return context.new_full((count, 10, 2), float(base_std))

        def sample_with_base(self, context, count, generator, base_std):
            plan = self.sample(context, count, generator, base_std)
            return plan, torch.zeros_like(plan)

        def embed(self, context, plan, base=None):
            value = plan.mean() if base is None else plan.mean() + base.mean()
            return torch.cat((context.reshape(1, -1), value.reshape(1, 1)), dim=1)

    fake = FakeDataset()
    observed = {}

    def fake_load_split(dataset, gammas, **kwargs):
        observed.update(dataset=dataset, gammas=tuple(gammas), kwargs=kwargs)
        return fake, object(), dict(
            manifest="/dataset/manifest.json", manifest_sha256="a" * 64,
            split_seed=LAUNCH.PRETRAIN.DEFAULT_SEED,
            split_semantics="fake globally disjoint split",
        )

    monkeypatch.setattr(LAUNCH.PRETRAIN, "load_split", fake_load_split)
    features, provenance = LAUNCH.calibration_features(
        FakeAdapter(), dataset_root="/dataset",
        expected_manifest_sha256="a" * 64, count=50, seed=2, base_std=1.0,
        paired_noised_representation=True,
    )
    assert features.shape == (50, 3)
    assert observed["kwargs"]["require_canonical"] is True
    assert observed["kwargs"]["expected_manifest_sha256"] == "a" * 64
    assert observed["kwargs"]["seed"] == LAUNCH.PRETRAIN.DEFAULT_SEED
    assert sum(provenance["per_gamma"].values()) == 50
    assert sorted(provenance["per_gamma"].values()) == [7, 7, 7, 7, 7, 7, 8]
    for gamma_index in range(len(SS.GAMMAS)):
        rows = [
            row for row in provenance["selection"]
            if row["gamma_index"] == gamma_index
        ]
        assert len({row["episode"] for row in rows}) == len(rows)
    assert provenance["manifest_sha256"] == "a" * 64
    assert provenance["split_seed"] == LAUNCH.PRETRAIN.DEFAULT_SEED
    assert provenance["selection_seed"] == 2
    assert provenance["paired_noised_representation"] is True


def test_production_scenarios_are_distinct_but_crn_pairable_across_gamma():
    task = PORT.SFMHP100ExpansionTask(scenario_start=300_000)
    first = task.reset(0.1, episode=0, seed=12345)
    paired = task.reset(1.0, episode=99, seed=12345)
    other = task.reset(0.1, episode=1, seed=12346)
    assert first.scenario_id == paired.scenario_id
    assert first.scenario_id != other.scenario_id
    assert first.scenario_id >= 300_000


def test_exact_verifier_uses_clipped_rollout_and_no_extra_progress_gate():
    task = PORT.SFMHP100ExpansionTask()
    robot = np.array([1.0, 1.0, DYN.V_MAX, 0.0], np.float32)
    ped_xy = np.full((task.n_ped, 2), 100.0, np.float32)
    ped_vel = np.zeros_like(ped_xy)
    context = _packed(task, robot, ped_xy, ped_vel)
    controls = torch.full((PORT.H, 2), DYN.U_MAX, dtype=torch.float32)
    result, sidecar = task._verify_one(context, controls, gamma=0.5)
    expected = PORT.clipped_plan_states(robot, controls.numpy())[:, :2]
    assert np.array_equal(sidecar["result"]["segment"], expected)
    assert result.progress_eligible is True
    assert sidecar["execution_eligible"] == bool(result.valid)
    assert sidecar["result"]["diagnostics"]["K_artificial"] == VERIFY.ARTIFICIAL_FACES
    assert len([
        face for face in sidecar["result"]["faces"]
        if face["kind"] == "artificial"
    ]) == 16


def test_terminal_suffix_never_weakens_full_h_positive_definition():
    task = PORT.SFMHP100ExpansionTask()
    robot = np.array([1.0, 1.0, 0.0, 0.0], np.float32)
    ped_xy = np.full((task.n_ped, 2), 100.0, np.float32)
    ped_vel = np.zeros_like(ped_xy)
    context = _packed(task, robot, ped_xy, ped_vel)
    short = torch.zeros((PORT.H - 1, 2), dtype=torch.float32)
    result, sidecar = task._verify_one(context, short, gamma=0.5)
    assert sidecar["result"]["full_h"] is False
    assert sidecar["result"]["taskspace"] is True
    assert sidecar["result"]["collision_free"] is True
    assert result.valid is False
    assert sidecar["execution_eligible"] is False


def test_protocol_archives_preterminal_queries_with_paired_x0():
    config = LAUNCH.protocol_config(_args(), lengthscale=0.25)
    assert (config.K, config.B) == (16, 4)
    assert config.parallel_episodes == 16
    assert config.archive_rule == "preterminal_resolved_queries"
    assert config.replay_acceptance == "safety_valid"
    assert config.successful_trajectories_per_gamma == 1
    assert config.execution_rule == "min_cost"
    assert config.execution_step_margin_weight == 70_000.0
    assert config.head_only_update and config.freeze_visual_encoder
    assert config.paired_noised_representation is True
    assert config.negative_alpha == 0.001
    assert config.ess_target == 0.1
    assert config.gp_reference_mode == "sliding_positive_per_gamma_frozen_phi"


def test_sfm_gp_cap_requires_equal_gamma_capacity():
    with pytest.raises(ValueError, match="divisible by seven"):
        LAUNCH.protocol_config(_args(gp_buffer_cap=768), lengthscale=0.25)


def test_microbatch_repeat_is_not_mislabeled_as_distinct_optimizer_cap():
    config = LAUNCH.protocol_config(
        _args(microbatch_repeats=10, optimizer_steps_per_round=None),
        lengthscale=0.25,
    )
    assert config.inner_steps is None
    assert config.microbatch_repeats == 10


def test_fixed_scenario_is_explicitly_audit_only():
    task = PORT.SFMHP100ExpansionTask(fixed_scenario_id=250_007)
    a = task.reset(0.1, 0, 1)
    b = task.reset(1.0, 1, 999)
    assert a.scenario_id == b.scenario_id == 250_007
    parsed = LAUNCH.parser().parse_args([
        "--mode", "run", "--checkpoint", "x",
        "--expected-checkpoint-sha256", "0" * 64,
        "--pretrain-dataset-root", "dataset",
        "--expected-pretrain-dataset-manifest-sha256", "1" * 64,
        "--output", "y", "--device", "cpu", "--physical-gpu", "-1",
        "--initial-beta", "0.01", "--audit-fixed-scenario-id", "250007",
    ])
    assert parsed.audit_fixed_scenario_id == 250_007


def test_paired_x0_cli_is_exposed_for_canonical_preterminal_run():
    parsed = LAUNCH.parser().parse_args([
        "--checkpoint", "x", "--expected-checkpoint-sha256", "0" * 64,
        "--pretrain-dataset-root", "dataset",
        "--expected-pretrain-dataset-manifest-sha256", "1" * 64,
        "--output", "y", "--device", "cpu", "--physical-gpu", "-1",
        "--initial-beta", "0.01", "--paired-noised-representation",
    ])
    assert parsed.paired_noised_representation is True


def test_score_log_is_flat_lambda_input_and_reports_per_gamma_ess(tmp_path):
    scores = tmp_path / "scores.jsonl"
    acquisition = tmp_path / "acquisition.jsonl"
    logger = LAUNCH.ScoreLog(scores, acquisition, 10.0)
    event = dict(
        round=2, gamma=0.5, context_id=7, episode=3,
        retry_batch=0, replica=1, step=4,
        selected=[9, 2, 5, 1], selected_sigma=[0.9, 0.8, 0.7, 0.6],
        sigma_K=[0.9, 0.6, 0.8, 0.5, 0.55, 0.7, *([0.4] * 10)],
        chosen_local=1, nvp_reason=None,
        verification=[
            dict(valid=True, error=False, progress_eligible=True,
                 target_eligible=True, execution_cost=10.0 + index,
                 step_margin=0.1 * index, progress=0.25 * index)
            for index in range(4)
        ],
    )
    logger(event)
    logger.close()
    rows = [json.loads(line) for line in scores.read_text().splitlines()]
    assert len(rows) == 4
    assert rows[0]["context_id"] == 7
    assert rows[0]["native_cost"] == 10.0
    assert rows[1]["chosen"] is True
    assert rows[1]["H10_goal_progress"] == 0.25
    assert rows[1]["archived_terminal_negative"] is False
    assert "B" not in rows[0]
    report = LAUNCH.acquisition_audit(
        acquisition, {"rounds": [{"round": 2, "beta": 0.05}]},
    )
    cell = report["cells"]["r002_g0.5"]
    assert cell["contexts"] == 1
    assert 0.0 < cell["marginal_ESS_over_K"] <= 1.0
    assert cell["uncertainty_uplift"] > 0.0


def test_generic_core_checkpoint_is_exported_for_canonical_raw_evaluator(tmp_path):
    policy = GPS.GridSFMHP100FlowPolicy()
    torch.save(dict(
        round=0, model=policy.state_dict(), config={"rounds": 10},
        pretrained=True,
    ), tmp_path / "checkpoint_000.pt")
    exported = LAUNCH.export_evaluation_checkpoints(
        tmp_path, architecture=policy.config(), pretrained_sha256="a" * 64,
    )
    row = exported["rows"][0]
    loaded, payload = GPS.load_sfm_hp100_policy(row["path"], device="cpu")
    assert payload["config"] == policy.config()
    assert payload["expansion_round"] == 0
    assert all(
        torch.equal(value, loaded.state_dict()[key])
        for key, value in policy.state_dict().items()
    )
