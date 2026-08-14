from dataclasses import dataclass
import json

import numpy as np
import pytest
import torch
from torch import nn

import sfm_hp100_early_acquisition as E
from sfm_hp100_ball_core.expansion import Verification


def _canonical_provenance_payload(manifest_sha="a" * 64):
    return {
        "scientific_status": "canonical_ID_promoted",
        "pretrained_from_scratch": True,
        "dataset": {"manifest_sha256": manifest_sha},
        "provenance": {"dataset_manifest_sha256": manifest_sha},
    }


def test_checkpoint_provenance_accepts_embedded_canonical_r0():
    resolved = E.resolve_checkpoint_provenance(
        _canonical_provenance_payload(), checkpoint_path="unused.pt",
        checkpoint_sha256="b" * 64,
        pretrained_provenance_checkpoint=None, expansion_delivery_marker=None,
    )
    assert resolved == {
        "mode": "embedded",
        "pretrain_dataset_manifest_sha256": "a" * 64,
        "pretrained_scientific_status": "canonical_ID_promoted",
    }


def test_checkpoint_provenance_authenticates_portable_expansion(tmp_path):
    anchor = tmp_path / "r0.pt"
    torch.save(_canonical_provenance_payload(), anchor)
    anchor_sha = E.APPROVAL.sha256_file(anchor)
    generic = tmp_path / "checkpoint_002.pt"
    torch.save({"model": {"head": torch.ones(1)}, "round": 2}, generic)
    generic_sha = E.APPROVAL.sha256_file(generic)
    expanded = tmp_path / "evaluation_checkpoint_002.pt"
    expanded_payload = {
        "scientific_status": "HP100_BALL_PORT_EVALUATION_READY",
        "pretrained_checkpoint_sha256": anchor_sha,
        "source_generic_checkpoint": str(generic),
        "source_generic_checkpoint_sha256": generic_sha,
    }
    torch.save(expanded_payload, expanded)
    expanded_sha = E.APPROVAL.sha256_file(expanded)
    marker = tmp_path / "EARLY_EXPANSION_COMPLETE.json"
    marker.write_text(json.dumps({
        "status": "SFM_HP100_EARLY_EXPANSION_COMPLETE",
        "preflight": {
            "checkpoint_sha256": anchor_sha,
            "calibration": {"manifest_sha256": "a" * 64},
        },
        "evaluation_checkpoints": {"rows": [{
            "path": str(expanded), "sha256": expanded_sha,
            "source_sha256": generic_sha,
        }]},
    }))

    resolved = E.resolve_checkpoint_provenance(
        expanded_payload, checkpoint_path=str(expanded),
        checkpoint_sha256=expanded_sha,
        pretrained_provenance_checkpoint=str(anchor),
        expansion_delivery_marker=str(marker),
    )
    assert resolved["mode"] == "anchored_expansion"
    assert resolved["pretrained_checkpoint_sha256"] == anchor_sha
    assert resolved["source_generic_checkpoint_sha256"] == generic_sha
    assert resolved["pretrain_dataset_manifest_sha256"] == "a" * 64


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 2)
        self.d = 20
        self.u_max = 2.0

    def sample(self, n, context, nfe, temp, initial_noise):
        del context, nfe, temp
        return initial_noise.reshape(n, 10, 2).clamp(-2.0, 2.0)

    def phi_s_from_x0(self, plans, context, base, s):
        del context, s
        return torch.stack((plans.reshape(len(plans), -1).mean(1),
                            base.reshape(len(base), -1).mean(1)), dim=1)


class _Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = _Policy()
        self.nfe = 1

    def _policy_context(self, context):
        return context.reshape(-1)[:1]

    def cfm_loss(self, *_args, **_kwargs):
        raise AssertionError("no-update diagnostic called cfm_loss")


@dataclass
class _State:
    scenario_id: int
    steps: int = 0
    status: str | None = None
    robot: np.ndarray | None = None


class _Task:
    def __init__(self, positive_steps=2):
        self.positive_steps = int(positive_steps)
        self.reset_rows = []
        self.scene_profile = "double_density_velocity_ood"
        self.profile = {
            "name": self.scene_profile, "n_ped": 40,
            "ped_speed_range": [1.0, 2.0],
        }
        self.scenario_start = 300_000
        self.scene_ledger = []

    def reset(self, gamma, episode, seed):
        self.reset_rows.append((float(gamma), int(episode), int(seed)))
        scenario_id = 300_000 + int(seed) % 1_000_000_000
        self.scene_ledger.append({
            "core_episode": int(episode), "scenario_id": scenario_id,
            "gamma": float(gamma), "reset_seed": int(seed),
            "fixed_scenario_audit": False,
        })
        return _State(
            scenario_id=scenario_id,
            robot=np.zeros(4, np.float32),
        )

    def context(self, state, gamma):
        return torch.tensor([float(state.steps), float(gamma)])

    def verify(self, context, candidates, gamma):
        del gamma
        positive = int(round(float(context[0]))) < self.positive_steps
        return tuple(
            Verification(
                valid=positive, hp_eligible=True, margin=1.0,
                execution_cost=float(index), progress_eligible=True,
                error=False, step_margin=float(index) / 10.0,
            )
            for index in range(len(candidates))
        )

    def decode_context(self, context):
        del context
        return np.zeros(4, np.float32), np.zeros((1, 2), np.float32), np.zeros((1, 2), np.float32)

    def advance(self, state, candidate):
        del candidate
        state.steps += 1
        state.robot[0] += .01
        return state

    def terminal(self, state):
        return state.status


def _run(task, *, K=2, B=1, std=1.0, event_callback=None):
    return E.run_diagnostic(
        _Adapter(), task, seeds=(2,), max_steps=30, K=K, B=B,
        flow_base_std=std, beta=1e-3, rbf_lengthscale=.5,
        rbf_noise=1e-2, execution_rule="max_step_margin",
        step_margin_weight=700_000.0, parallel_episodes=16,
        verifier_workers=1, audit_unselected_at_nvp=True,
        event_callback=event_callback,
    )


def test_no_update_and_paired_16_by_7_lineages_with_nvp_accounting():
    payload, queries, contexts = _run(_Task(positive_steps=2))
    assert payload["policy_unchanged"] is True
    assert payload["policy_state_sha256_before"] == payload["policy_state_sha256_after"]
    assert len(payload["lineages"]) == 16 * 7
    assert {row["status"] for row in payload["lineages"]} == {"nvp"}
    assert {row["nvp_step"] for row in payload["lineages"]} == {2}
    assert {row["executed_steps"] for row in payload["lineages"]} == {2}
    assert len(queries) == 16 * 7 * 3
    assert len(contexts) == 16 * 7 * 3
    assert payload["summary"]["pooled"]["selected_B"]["Dplus"] == 16 * 7 * 2
    assert payload["summary"]["pooled"]["selected_B"]["Dminus"] == 16 * 7
    assert payload["summary"]["pooled"]["early_gather_RMST_at_30"] == 2.0
    assert payload["summary"]["pooled"][
        "lineage_macro_net_goal_progress_at_30"
    ] > 0.0
    assert payload["summary"]["pooled"]["goal_progress"][
        "chosen_H10_progress_percentile_mean_at_30"
    ] == 1.0
    assert set(payload["summary"]["per_seed"]) == {"2"}
    for replica in range(16):
        ids = {
            row["scenario_id"] for row in payload["lineages"]
            if row["replica"] == replica
        }
        assert len(ids) == 1


def test_max_step_margin_uses_native_cost_only_as_tie_break():
    results = [
        Verification(True, True, 1.0, 0.0, step_margin=.1),
        Verification(True, True, 1.0, 100.0, step_margin=.2),
        Verification(True, True, 1.0, -100.0, step_margin=.2),
    ]
    chosen, scores = E._select(
        results, rule="max_step_margin", step_margin_weight=0.0,
    )
    assert chosen == 2
    assert scores == [-.1, -.2, -.2]


def test_trace_event_distinguishes_executed_positive_from_terminal_counterfactual():
    events = []
    _run(_Task(positive_steps=1), event_callback=events.append)
    first = next(row for row in events if row["gamma"] == .1 and row["replica"] == 0)
    terminal = next(
        row for row in events
        if row["gamma"] == .1 and row["replica"] == 0 and row["step"] == 1
    )
    assert first["chosen_local"] == 0
    assert first["archived_negative_local"] is None
    assert first["verification"][0]["valid"] is True
    assert terminal["chosen_local"] is None
    assert terminal["archived_negative_local"] == 0
    assert terminal["verification"][0]["valid"] is False
    assert terminal["status"] == "nvp"
    assert terminal["queried_segments"].shape == (1, 11, 2)


def test_terminal_negative_uses_the_same_weighted_score_as_production():
    events = []
    _run(_Task(positive_steps=1), K=2, B=2, event_callback=events.append)
    terminal = next(
        row for row in events
        if row["gamma"] == .1 and row["replica"] == 0 and row["step"] == 1
    )
    # Native costs are [0,1], margins are [0,.1], and lambda=700k.
    assert terminal["archived_negative_local"] == 1


def test_progress_rank_and_lambda_zero_reference_are_explicit():
    events = []
    _, queries, contexts = _run(
        _Task(positive_steps=1), K=2, B=2, event_callback=events.append,
    )
    first = next(
        row for row in contexts
        if row["gamma"] == .1 and row["replica"] == 0 and row["step"] == 0
    )
    # Max-margin chooses slot 1, while the lambda-zero native-cost reference is slot 0.
    assert first["chosen_H10_progress_rank"] == 2
    assert first["chosen_H10_progress_percentile"] == 0.0
    chosen = next(
        row for row in queries
        if row["gamma"] == .1 and row["replica"] == 0
        and row["step"] == 0 and row["chosen"]
    )
    reference = next(
        row for row in queries
        if row["gamma"] == .1 and row["replica"] == 0
        and row["step"] == 0 and row["performance_reference"]
    )
    assert chosen["selected_slot"] == 1
    assert reference["selected_slot"] == 0
    trace = next(
        row for row in events
        if row["gamma"] == .1 and row["replica"] == 0 and row["step"] == 0
    )
    assert trace["chosen_H10_progress_rank"] == 2
    assert trace["eligible_progress_candidates"] == 2
    assert all(row["progress_eligible"] for row in trace["verification"])


def test_trace_marks_step_30_as_early_cutoff_not_active():
    events = []
    _run(_Task(positive_steps=100), event_callback=events.append)
    final = next(
        row for row in events
        if row["gamma"] == .1 and row["replica"] == 0 and row["step"] == 29
    )
    assert final["status"] == "EARLY_CUTOFF"


def test_retained_blue_branch_gets_exact_16_face_sidecar_only_after_filtering():
    task = _Task(positive_steps=1)
    events = []
    _run(task, event_callback=events.append)
    event = next(row for row in events if row["gamma"] == .1 and row["replica"] == 0)
    calls = []

    def verify_one(context, candidate, gamma):
        calls.append((context, candidate, gamma))
        result = task.verify(context, candidate[None], gamma)[0]
        segment = E.PORT.clipped_plan_states(
            np.zeros(4, np.float32), candidate.numpy(),
        )[:, :2]
        faces = [
            dict(
                a=[np.cos(2 * np.pi * index / 16), np.sin(2 * np.pi * index / 16)],
                m=2.0, kind="artificial", label=f"art{index}",
                coefficient=1.0, feasible=True, interval=None,
            )
            for index in range(16)
        ]
        return result, dict(result=dict(
            resolved=True, y=1, full_h=True, segment=segment, faces=faces,
            diagnostics=dict(
                solver="paper_static_exact_2d_angular_interval_socp",
                K_artificial=16,
            ),
        ))

    task._verify_one = verify_one
    attached = E._attach_exact_chosen_sidecar(task, event)
    assert len(calls) == 1
    assert "context" not in attached
    assert len(attached["chosen_verifier_sidecar"]["faces"]) == 16
    assert attached["chosen_verifier_sidecar"]["pedestrian_prediction"].shape == (
        11, 1, 2,
    )


def test_batched_sampling_preserves_base_std_scaling():
    adapter = _Adapter()
    contexts = [torch.tensor([0.0, .1]), torch.tensor([0.0, 1.0])]
    _, base1, _, _ = E._sample_blocks(
        adapter, contexts, (11, 12), K=4, flow_base_std=1.0,
    )
    _, base2, _, _ = E._sample_blocks(
        adapter, contexts, (11, 12), K=4, flow_base_std=2.0,
    )
    assert torch.equal(base2[0], 2.0 * base1[0])
    assert torch.equal(base2[1], 2.0 * base1[1])


def test_same_lineage_seed_pairs_base_noise_exactly_across_gamma():
    adapter = _Adapter()
    contexts = [torch.tensor([0.0, .1]), torch.tensor([0.0, 1.0])]
    _, bases, _, _ = E._sample_blocks(
        adapter, contexts, (77, 77), K=8, flow_base_std=2.0,
    )
    assert torch.equal(bases[0], bases[1])


@pytest.mark.parametrize("K,B", [(0, 1), (2, 0), (2, 3)])
def test_invalid_candidate_budgets_fail(K, B):
    with pytest.raises(ValueError):
        E.run_diagnostic(
            _Adapter(), _Task(), seeds=(2,), max_steps=30, K=K, B=B,
            flow_base_std=1.0, beta=1e-3, rbf_lengthscale=.5,
            rbf_noise=1e-2, execution_rule="max_step_margin",
            step_margin_weight=0.0, parallel_episodes=16,
            verifier_workers=1, audit_unselected_at_nvp=False,
        )


def test_parallel_episode_contract_is_fixed_at_16():
    with pytest.raises(ValueError, match="exactly 16"):
        E.run_diagnostic(
            _Adapter(), _Task(), seeds=(2,), max_steps=30, K=2, B=1,
            flow_base_std=1.0, beta=1e-3, rbf_lengthscale=.5,
            rbf_noise=1e-2, execution_rule="max_step_margin",
            step_margin_weight=0.0, parallel_episodes=8,
            verifier_workers=1, audit_unselected_at_nvp=False,
        )
