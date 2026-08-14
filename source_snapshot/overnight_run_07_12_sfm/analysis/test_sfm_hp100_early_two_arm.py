from argparse import Namespace
import copy
import hashlib
import json

import pytest

import sfm_hp100_early_two_arm as RUN


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selection(checkpoint_sha, *, status=RUN.SELECTION_STATUS, selected=True):
    return {
        "status": status,
        "selection_blind_to_training_and_evaluation": True,
        "checkpoint_sha256": checkpoint_sha,
        "selected": {"lambda": 30_000.0, "passes": True} if selected else None,
    }


def _args(tmp_path):
    return Namespace(
        source_root=str(tmp_path / "source"), checkpoint=str(tmp_path / "r0.pt"),
        expected_checkpoint_sha256="a" * 64,
        pretrain_dataset_root=str(tmp_path / "data"),
        expected_pretrain_dataset_manifest_sha256="b" * 64,
        output_root=str(tmp_path / "out"), scene_profile="double_density_velocity_ood",
        scenario_start=400_000, rounds=3, verifier_workers=32, batch_size=64,
        learning_rate=1e-6, microbatch_repeats=10, flow_base_std=1.4,
        negative_alpha=1e-3, seed=2,
    )


def _options(command):
    values = {}
    index = 3
    while index < len(command):
        values[command[index]] = command[index + 1]
        index += 2
    return values


def test_lambda_report_is_authenticated_and_fail_closed(tmp_path):
    checkpoint_sha = "c" * 64
    path = tmp_path / "lambda.json"
    path.write_text(json.dumps(_selection(checkpoint_sha)))
    report, value = RUN.load_selected_lambda(
        path, expected_sha256=_sha(path), checkpoint_sha256=checkpoint_sha,
    )
    assert report["selected"]["passes"] is True
    assert value == 30_000.0

    path.write_text(json.dumps(_selection(checkpoint_sha, selected=False)))
    with pytest.raises(ValueError, match="fail-closed"):
        RUN.load_selected_lambda(
            path, expected_sha256=_sha(path), checkpoint_sha256=checkpoint_sha,
        )


def test_two_arms_have_disjoint_gpus_and_only_scope_specific_commands(tmp_path):
    arms = RUN.declared_arms((1, 3))
    assert [(arm.optimizer_scope, arm.physical_gpu) for arm in arms] == [
        ("head_only", 1), ("last_block_and_head", 3),
    ]
    args = _args(tmp_path)
    commands = [
        RUN.expansion_command(args, arm, selected_lambda=30_000.0, mode="run")
        for arm in arms
    ]
    left, right = map(_options, commands)
    assert left["--optimizer-scope"] == "head_only"
    assert right["--optimizer-scope"] == "last_block_and_head"
    for differing in ("--output", "--physical-gpu", "--optimizer-scope"):
        left.pop(differing); right.pop(differing)
    assert left == right
    assert left["--flow-base-std"] == "1.4"
    assert left["--execution-step-margin-weight"] == "30000.0"
    assert left["--seed"] == "2"


def test_preflight_signature_ignores_only_optimizer_scope():
    common = {
        "version": "v", "checkpoint_sha256": "a" * 64,
        "architecture": {"name": "hp100"}, "scene_profile": {"n_ped": 40},
        "scenario_bank": {"start": 400_000},
        "calibration": {"lengthscale": .2},
        "expansion_config": {
            "optimizer_scope": "head_only", "flow_base_std": 1.4,
            "execution_step_margin_weight": 30_000.0, "seed": 2,
        },
    }
    other = copy.deepcopy(common)
    other["expansion_config"]["optimizer_scope"] = "last_block_and_head"
    assert RUN.shared_preflight_signature(common) == RUN.shared_preflight_signature(other)
    other["expansion_config"]["seed"] = 3
    assert RUN.shared_preflight_signature(common) != RUN.shared_preflight_signature(other)


def test_exactly_two_distinct_gpus_are_required():
    with pytest.raises(ValueError, match="two distinct"):
        RUN.declared_arms((1, 1))
    with pytest.raises(ValueError, match="two distinct"):
        RUN.declared_arms((1,))
