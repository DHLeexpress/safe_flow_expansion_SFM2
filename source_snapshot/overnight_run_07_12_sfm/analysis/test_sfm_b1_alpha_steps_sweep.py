import json

import pytest

import sfm_b1_alpha_steps_sweep as S


def _cell(value):
    return dict(
        n=50, SR=1-value, CR=value, V_safe=1-value, timeout=0.0,
        verifier_errors=0,
        successful_clearance=dict(mean=.2), successful_time_to_goal=dict(mean=8.0),
    )


def _record(value):
    return dict(
        round=2,
        temperature_by_gamma={str(g): 1.0 for g in S.CE.SP.GAMMAS},
        summary=dict(pooled=_cell(value), per_gamma={str(g): _cell(value) for g in S.CE.SP.GAMMAS}),
    )


def test_factorial_is_exact_nine_margin_arms():
    arms = S.arm_grid()
    assert len(arms) == 9 and len({arm.name for arm in arms}) == 9
    assert {(arm.alpha, arm.inner_epochs) for arm in arms} == {
        (alpha, epochs) for alpha in S.ALPHAS for epochs in S.INNER_EPOCHS
    }
    assert arms[0].name == "margin_alpha0_inner001"


def test_slot_waves_schedule_eight_then_one(monkeypatch, tmp_path):
    waves = []

    def fake_run_parallel(jobs, logdir):
        waves.append(list(jobs))
        return [str(tmp_path / f"{name}.log") for _, _, name in jobs]

    monkeypatch.setattr(S.SW, "run_parallel", fake_run_parallel)
    jobs = [(["python", f"{index}.py"], f"job{index}") for index in range(9)]
    logs = S._slot_waves(jobs, tmp_path)
    assert [len(wave) for wave in waves] == [8, 1]
    assert [slot for slot, _, _ in waves[0]] == list(S.SCHEDULER_SLOTS)
    assert waves[1][0][0] == S.SCHEDULER_SLOTS[0]
    assert len(logs) == 9


def test_screening_prefers_metrics_before_update_complexity():
    simple = S.SweepArm(0.0, 1)
    strong = S.SweepArm(.01, 16)
    assert S.screening_key(_record(.1), strong) < S.screening_key(_record(.2), simple)
    assert S.screening_key(_record(.1), simple) < S.screening_key(_record(.1), strong)


def test_preflight_is_bound_to_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "r0.pt"; checkpoint.write_bytes(b"checkpoint")
    payload = dict(
        status="RBF_PREFLIGHT_COMPLETE", checkpoint_sha256=S.SW.sha256_file(checkpoint),
        lengthscale_count=50, lambda_=1e-2,
        selected=dict(
            ess_solved=True, stable_conditioning=True, cap=256, ell=.2,
            ell_multiplier=.5,
        ),
        candidates=[dict(
            ess_solved=True, stable_conditioning=True, cap=512, ell=.2,
            ell_multiplier=.5,
        )],
    )
    path = tmp_path / "preflight.json"; path.write_text(json.dumps(payload))
    expected_sha = S.SW.sha256_file(path)
    loaded = S._load_preflight(path, checkpoint, expected_sha)
    assert loaded["sweep_selected"] == payload["candidates"][0]
    try:
        S._load_preflight(path, checkpoint, "bad")
        assert False
    except RuntimeError as error:
        assert "preflight SHA-256" in str(error)
    payload["checkpoint_sha256"] = "bad"; path.write_text(json.dumps(payload))
    try:
        S._load_preflight(path, checkpoint, S.SW.sha256_file(path))
        assert False
    except RuntimeError as error:
        assert "checkpoint" in str(error)


def test_output_root_is_fail_closed_and_symlink_safe(tmp_path, monkeypatch):
    root = tmp_path / "research1"
    root.mkdir()
    monkeypatch.setattr(S, "OUTPUT_ROOT", str(root))
    assert S._validate_output_root(root / "fresh") == str(root / "fresh")
    with pytest.raises(ValueError, match="fresh directory"):
        S._validate_output_root(root)
    with pytest.raises(ValueError, match="fresh directory"):
        S._validate_output_root(tmp_path / "elsewhere")


def test_development_shortlist_keeps_best_per_alpha_plus_best_remaining(tmp_path):
    arms = S.arm_grid()
    # E=1 wins within every alpha; alpha=0/E=4 is the best remaining arm.
    arm_values = {
        arm: (.10 + arm.alpha + {1: 0.0, 4: .01, 16: .03}[arm.inner_epochs])
        for arm in arms
    }
    temperatures = {str(gamma): 1.0 for gamma in S.CE.SP.GAMMAS}
    for arm in arms:
        directory = tmp_path / "arms" / arm.name
        directory.mkdir(parents=True)
        baseline = dict(
            round=0, temperature_by_gamma=temperatures,
            summary=_record(arm_values[arm] + .2)["summary"],
        )
        adapted = dict(
            round=1, temperature_by_gamma=temperatures,
            summary=_record(arm_values[arm])["summary"],
        )
        (directory / "method_manifest.json").write_text(json.dumps(dict(
            status="ARM_COMPLETE", rounds=1,
            baseline_sanity=baseline, history=[dict(sanity=adapted)],
        )))
    _, shortlist = S._development_shortlist(tmp_path, arms, expected_rounds=1)
    selected = [arm for arm, _ in shortlist]
    assert len(selected) == len(set(selected)) == 4
    assert {(arm.alpha, arm.inner_epochs) for arm in selected[:3]} == {
        (alpha, 1) for alpha in S.ALPHAS
    }
    assert (selected[3].alpha, selected[3].inner_epochs) == (0.0, 4)


def test_runtime_forecast_and_logs_must_stay_under_output_root(tmp_path, monkeypatch):
    root = tmp_path / "research1"
    root.mkdir()
    monkeypatch.setattr(S, "OUTPUT_ROOT", str(root))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    log = root / "gate.log"
    log.write_text("complete")
    source = dict(commit="abc")
    args = type("Args", (), dict(
        checkpoint=str(checkpoint), scene_profile="double_density_velocity_ood",
        rounds=20, max_hours=6.0,
    ))()
    payload = dict(
        status="RUNTIME_GATE_PASS", source_commit="abc",
        checkpoint_sha256=S.SW.sha256_file(checkpoint), preflight_sha256="preflight",
        scene_profile=args.scene_profile, workers_per_arm=S.ARM_WORKERS,
        arm_count=len(S.arm_grid()), parallel_slots=list(S.SCHEDULER_SLOTS),
        rounds=20, benchmark_rounds=S.RUNTIME_GATE_ROUNDS,
        forecast_seconds=5.0 * 3600.0, limit_seconds=6.0 * 3600.0,
        logs=[str(log)],
    )
    forecast = root / "forecast.json"
    forecast.write_text(json.dumps(payload))
    assert S._load_runtime_forecast(
        forecast, source=source, args=args, preflight_sha256="preflight",
    ) == payload

    payload["parallel_slots"] = ["1a", "3a", "1b", "3b"]
    forecast.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="eight-slot"):
        S._load_runtime_forecast(
            forecast, source=source, args=args, preflight_sha256="preflight",
        )
    payload["parallel_slots"] = list(S.SCHEDULER_SLOTS)

    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="stored under"):
        S._load_runtime_forecast(
            outside, source=source, args=args, preflight_sha256="preflight",
        )

    payload["logs"] = [str(tmp_path / "missing.log")]
    forecast.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="log must exist"):
        S._load_runtime_forecast(
            forecast, source=source, args=args, preflight_sha256="preflight",
        )

    payload["logs"] = [str(log)]
    payload["forecast_seconds"] = 7.0 * 3600.0
    forecast.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="current time limit"):
        S._load_runtime_forecast(
            forecast, source=source, args=args, preflight_sha256="preflight",
        )
