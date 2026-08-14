import ast
import json
from pathlib import Path

import pytest

import sfm_b1_sharded_benchmark as S
import sfm_protocol as SP
import sfm_scene as SS


def _base(tmp_path, M=2, scene_profile="double_density_velocity_ood"):
    r0 = tmp_path / "r0.pt"
    selected = tmp_path / "selected.pt"
    r0.write_bytes(b"r0")
    selected.write_bytes(b"selected")
    gpu = dict(
        status="SFM_B1_GPU_PROVENANCE", declared_cuda_visible_device="3",
        index="3", uuid="GPU-test", name="H100", driver_version="1",
        source=dict(commit="a" * 40, tracked_worktree_clean=True),
    )
    return dict(
        source=dict(commit="a" * 40, tracked_worktree_clean=True),
        checkpoints={
            "r0": dict(path=str(r0.resolve()), sha256=S.BE.sha256_file(r0)),
            "selected": dict(path=str(selected.resolve()), sha256=S.BE.sha256_file(selected)),
        },
        scene_profile=scene_profile, environment=SS.scene_profile(scene_profile),
        ep0=SP.DEPLOY_DOUBLE_SHIFT_EP0, M_per_gamma=M, gpu=gpu,
    )


def _row(episode, gamma, *, success):
    return dict(
        episode=int(episode), gamma=float(gamma), success=bool(success),
        collision=bool(not success), reached=bool(success), timeout=False,
        steps=10, time_to_goal=(1.0 if success else None), min_clearance=.1,
        successful_clearance=(.1 if success else None), mode_counts={},
    )


def _payload(base, method, gamma):
    contract = S._cell_contract(base, method, gamma)
    checkpoint = base["checkpoints"][contract["used_checkpoint"]]
    rows = [_row(episode, gamma, success=(episode + METHODS_INDEX[method]) % 2 == 0)
            for episode in contract["episodes"]]
    result = dict(
        method=S.METHOD_RESULT_NAMES[method], checkpoint=checkpoint["path"],
        checkpoint_sha256=checkpoint["sha256"], summary={}, rows=rows,
    )
    if method in S.KAZUKI_CONFIGS:
        result.update(**S.KAZUKI_CONFIGS[method],
                      refinement_cost="b1_safemppi",
                      refinement_cost_manifest=S.BB.KZ.BC.scorer_manifest(),
                      comparator_semantics="learned prior plus reward guidance and MPPI refinement; not raw flow")
    else:
        result["raw_semantics"] = (
            "temp=1,NFE=8,one generated window per context,execute first action; no tilt/verifier/selector"
        )
    return dict(status="SFM_B1_BENCHMARK_CELL_COMPLETE", contract=contract, result=result)


METHODS_INDEX = {method: index for index, method in enumerate(S.METHODS)}


def test_cell_runs_only_declared_gamma_and_publishes_atomically(tmp_path, monkeypatch):
    base = _base(tmp_path)
    seen = {}

    def fake_evaluate(checkpoint, episodes, gamma, *, scene_profile, device):
        seen.update(
            checkpoint=checkpoint, episodes=list(episodes), gamma=gamma,
            scene_profile=scene_profile, device=device,
        )
        contract = S._cell_contract(base, "selected_raw", .3)
        chosen = base["checkpoints"]["selected"]
        return dict(
            method=S.METHOD_RESULT_NAMES["selected_raw"], checkpoint=chosen["path"],
            checkpoint_sha256=chosen["sha256"], raw_semantics="raw",
            summary={}, rows=[_row(episode, .3, success=True) for episode in contract["episodes"]],
        )

    monkeypatch.setattr(S, "_base_contract", lambda **kwargs: base)
    monkeypatch.setattr(S, "_evaluate_raw_cell", fake_evaluate)
    result = S.run_cell(
        r0="r0", selected="selected", scene_profile=base["scene_profile"],
        ep0=base["ep0"], M=2,
        method="selected_raw", gamma="0.3", device="cuda:0", outdir=tmp_path,
        expected_source_commit="a" * 40, expected_r0_sha256="r", expected_selected_sha256="s",
        expected_gpu_uuid="GPU-test",
    )
    assert result["contract"]["gamma"] == .3
    assert seen["episodes"] == [
        SP.DEPLOY_DOUBLE_SHIFT_EP0, SP.DEPLOY_DOUBLE_SHIFT_EP0 + 1,
    ] and seen["gamma"] == .3
    output = Path(S.cell_path(tmp_path, "selected_raw", .3))
    assert output.exists()
    assert not list(output.parent.glob("*.tmp.*"))
    assert json.loads(output.read_text())["status"] == "SFM_B1_BENCHMARK_CELL_COMPLETE"


def test_aggregate_requires_exact_28_unique_authenticated_cells(tmp_path, monkeypatch):
    base = _base(tmp_path)
    cells = tmp_path / "cells"
    cells.mkdir()
    for method in S.METHODS:
        for gamma in SP.GAMMAS:
            S._write_json(cells / S.cell_filename(method, gamma), _payload(base, method, gamma))
    S._write_json(tmp_path / "gpu_provenance.json", base["gpu"])

    monkeypatch.setattr(S, "_base_contract", lambda **kwargs: base)

    def fake_render(payload, png, csv):
        Path(png).write_bytes(b"png")
        Path(csv).write_text("csv")

    monkeypatch.setattr(S.BB, "_render_benchmark", fake_render)
    values = dict(
        r0="r0", selected="selected", scene_profile=base["scene_profile"],
        ep0=base["ep0"], M=2,
        outdir=tmp_path, expected_source_commit="a" * 40,
        expected_r0_sha256="r", expected_selected_sha256="s",
        expected_gpu_uuid="GPU-test",
    )
    payload = S.aggregate(**values)
    assert list(payload["methods"]) == [S.METHOD_LABELS[value] for value in S.METHODS]
    assert len(payload["cell_order"]) == 28
    assert payload["cell_order"][0] == dict(
        method="r0_raw", gamma=.1, file=S.cell_filename("r0_raw", .1),
    )
    assert json.loads((tmp_path / "COMPLETE.json").read_text())["cell_count"] == 28

    (cells / "duplicate.json").write_text((cells / S.cell_filename("r0_raw", .1)).read_text())
    with pytest.raises(RuntimeError, match="exactly 28"):
        S.aggregate(**values)


def test_scenario_cluster_bootstrap_preserves_all_gamma_rows_and_is_deterministic():
    rows = []
    for gamma in SP.GAMMAS:
        rows.extend([
            _row(10, gamma, success=True),
            _row(11, gamma, success=False),
        ])
    first = S.scenario_cluster_bootstrap(rows, seed=7, draws=4000)
    second = S.scenario_cluster_bootstrap(rows, seed=7, draws=4000)
    assert first == second
    assert first["rows_per_cluster"] == 7 and first["clusters"] == 2
    assert first["SR"] == dict(estimate=.5, interval95=[0.0, 1.0])
    assert first["CR"] == dict(estimate=.5, interval95=[0.0, 1.0])


def test_cell_rejects_wrong_contract_before_reuse(tmp_path):
    base = _base(tmp_path)
    payload = _payload(base, "r0_raw", .1)
    wrong = S._cell_contract(base, "r0_raw", .2)
    with pytest.raises(RuntimeError, match="contract mismatch"):
        S._validate_cell(payload, wrong)


def test_driver_declares_one_gpu_and_cells_are_subprocess_isolated(tmp_path):
    assert S._driver_environment("3")["CUDA_VISIBLE_DEVICES"] == "3"
    with pytest.raises(ValueError, match="exactly one"):
        S._driver_environment("1,3")
    args = S.build_parser().parse_args([
        "driver", "--r0", "r0", "--selected", "selected", "--outdir", str(tmp_path),
        "--scene-profile", "double_density_velocity_ood",
        "--expected-source-commit", "a" * 40, "--expected-r0-sha256", "r",
        "--expected-selected-sha256", "s", "--expected-gpu-uuid", "GPU-test",
        "--cuda-visible-device", "3",
    ])
    command = S._cell_command(args, "kazuki_goal_stress", .5)
    assert command[1:3] == [str(Path(S.__file__).resolve()), "cell"]
    assert command[-6:] == [
        "--method", "kazuki_goal_stress", "--gamma", "0.5", "--device", "cuda:0",
    ]
    source = Path(S.__file__).read_text()
    imports = [node for node in ast.walk(ast.parse(source))
               if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all("thread" not in ast.unparse(node).lower() for node in imports)


def test_gpu_provenance_resolves_declared_index_and_rejects_wrong_uuid(monkeypatch):
    output = (
        "1, GPU-one, H100 NVL, 550.1\n"
        "3, GPU-three, H100 NVL, 550.1\n"
    )
    monkeypatch.setattr(S.subprocess, "check_output", lambda *args, **kwargs: output)
    source = dict(commit="a" * 40, tracked_worktree_clean=True)
    result = S._gpu_snapshot("3", "GPU-three", source)
    assert (result["index"], result["uuid"], result["driver_version"]) == (
        "3", "GPU-three", "550.1",
    )
    with pytest.raises(RuntimeError, match="UUID mismatch"):
        S._gpu_snapshot("3", "GPU-one", source)


@pytest.mark.parametrize("profile", S.PROFILES)
def test_fixed_bank_accepts_both_profiles_with_shared_episode_range(profile):
    S._validate_fixed_bank(profile, SP.DEPLOY_DOUBLE_SHIFT_EP0, 100)


def test_fixed_bank_rejects_an_alternate_episode_range():
    with pytest.raises(ValueError, match="fixed to ep0"):
        S._validate_fixed_bank(
            "double_density_velocity_ood", SP.DEPLOY_DOUBLE_SHIFT_EP0 + 1, 100,
        )


def test_kazuki_arms_are_predeclared_and_distinct():
    assert S.KAZUKI_CONFIGS == {
        "kazuki_default": dict(safe_coef=.3, goal_coef=.5),
        "kazuki_goal_stress": dict(safe_coef=.3, goal_coef=1.0),
    }
    assert S.METHODS == (
        "r0_raw", "selected_raw", "kazuki_default", "kazuki_goal_stress",
    )


def test_goal_stress_cell_uses_and_authenticates_declared_coefficients(tmp_path, monkeypatch):
    checkpoint = tmp_path / "r0.pt"
    checkpoint.write_bytes(b"r0")
    seen = {}
    monkeypatch.setattr(S.BB.GPS, "load_sfm_policy", lambda *args, **kwargs: (object(), {}))

    def fake_deploy(policy, episode, gamma, *, cfg, **kwargs):
        seen.update(safe_coefs=cfg.safe_coefs, goal_coef=cfg.goal_coef)
        return dict(
            success=True, collision=False, reached=True, steps=3, min_clear=.2,
        )

    monkeypatch.setattr(S.BB.KZ, "kazuki_sfm_deploy", fake_deploy)
    result = S._evaluate_kazuki_cell(
        checkpoint, [SP.DEPLOY_DOUBLE_SHIFT_EP0], .1,
        method="kazuki_goal_stress", scene_profile="double_density_velocity_ood",
        device="cuda:0",
    )
    assert seen == dict(safe_coefs=(.3,), goal_coef=1.0)
    assert (result["safe_coef"], result["goal_coef"]) == (.3, 1.0)

    contract_dir = tmp_path / "contract"
    contract_dir.mkdir()
    base = _base(contract_dir)
    payload = _payload(base, "kazuki_goal_stress", .1)
    payload["result"]["goal_coef"] = .5
    with pytest.raises(RuntimeError, match="configuration mismatch"):
        S._validate_cell(payload, S._cell_contract(base, "kazuki_goal_stress", .1))
