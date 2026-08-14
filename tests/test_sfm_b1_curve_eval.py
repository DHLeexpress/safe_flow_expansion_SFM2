import copy
import json

import numpy as np
import pytest
import torch

import sfm_b1_curve_eval as C


def _cell(*, cr=.2, v_safe=.4, sr=.7, timeout=.1, clearance=.3, time=8.0):
    return dict(
        n=10, CR=cr, V_safe=v_safe, SR=sr, timeout=timeout,
        verifier_errors=0,
        successful_clearance=dict(mean=clearance),
        successful_time_to_goal=dict(mean=time),
    )


def _summary(**kwargs):
    values = {str(gamma): _cell(**kwargs) for gamma in C.SP.GAMMAS}
    return dict(pooled=_cell(**kwargs), per_gamma=values)


def test_banks_must_be_disjoint_and_nonempty():
    C.assert_disjoint_banks(100, 10, 200, 50)
    with pytest.raises(ValueError):
        C.assert_disjoint_banks(100, 10, 105, 50)
    with pytest.raises(ValueError):
        C.assert_disjoint_banks(100, 0, 200, 50)
    C.assert_final_confirmation_bank(C.SP.FINAL_CONFIRM_EP0, 100)
    with pytest.raises(ValueError):
        C.assert_final_confirmation_bank(C.SCREEN_EP0, 100)


def test_verifier_errors_abort_before_selection():
    clean = _summary()
    C.assert_zero_verifier_errors(clean, role="test")
    contaminated = copy.deepcopy(clean)
    contaminated["per_gamma"]["0.1"]["verifier_errors"] = 1
    with pytest.raises(RuntimeError, match="verifier errors"):
        C.assert_zero_verifier_errors(contaminated, role="test")


def test_temperature_vector_ties_prefer_one_and_can_differ_by_gamma():
    candidates = {temperature: _summary() for temperature in C.TEMPERATURES}
    chosen, key = C.select_temperature_vector(candidates)
    assert set(chosen) == {str(gamma) for gamma in C.SP.GAMMAS}
    assert set(chosen.values()) == {1.0} and isinstance(key, list)
    better = copy.deepcopy(candidates)
    better[.95]["per_gamma"]["0.1"]["CR"] = .1
    selected, _ = C.select_temperature_vector(better)
    assert selected["0.1"] == .95
    assert all(selected[str(gamma)] == 1.0 for gamma in C.SP.GAMMAS if gamma != .1)
    with pytest.raises(ValueError):
        C.select_temperature_vector({1.0: _summary()})
    with pytest.raises(ValueError, match="every declared gamma"):
        C.temperature_vector({"0.1": 1.0})


def test_selection_key_penalizes_gamma_order_violation():
    good = _summary()
    bad = copy.deepcopy(good)
    # Desired low-to-high gamma trend is non-increasing clearance and time.
    for index, gamma in enumerate(C.SP.GAMMAS):
        good["per_gamma"][str(gamma)]["successful_clearance"]["mean"] = 1.0 - index / 10
        good["per_gamma"][str(gamma)]["successful_time_to_goal"]["mean"] = 10.0 - index / 10
        bad["per_gamma"][str(gamma)]["successful_clearance"]["mean"] = 0.1 + index / 10
        bad["per_gamma"][str(gamma)]["successful_time_to_goal"]["mean"] = 5.0 + index / 10
    assert C.temperature_selection_key(good, 1.0) < C.temperature_selection_key(bad, 1.0)


def test_selection_key_is_finite_when_a_gamma_has_no_successes():
    summary = _summary()
    summary["per_gamma"]["0.1"]["successful_clearance"]["mean"] = None
    summary["per_gamma"]["0.1"]["successful_time_to_goal"]["mean"] = None
    key = C.temperature_selection_key(summary, 1.0)
    assert all(np.isfinite(float(value)) for value in key)


def test_cell_contract_key_authenticates_temperature_and_bank():
    bank = dict(version="x", ep0=1, M=10, sha256="a")
    base = C.cell_contract_key(
        checkpoint_sha256="c", round_i=2, scene_profile="matched_id",
        bank=bank, temperature=1.0, role="screen",
    )
    assert base == C.cell_contract_key(
        checkpoint_sha256="c", round_i=2, scene_profile="matched_id",
        bank=copy.deepcopy(bank), temperature=1.0, role="screen",
    )
    assert base != C.cell_contract_key(
        checkpoint_sha256="c", round_i=2, scene_profile="matched_id",
        bank=bank, temperature=.95, role="screen",
    )
    vector = C.temperature_vector(1.0)
    vector["0.1"] = .95
    assert base != C.cell_contract_key(
        checkpoint_sha256="c", round_i=2, scene_profile="matched_id",
        bank=bank, temperature=vector, role="screen",
    )
    changed = dict(bank, sha256="b")
    assert base != C.cell_contract_key(
        checkpoint_sha256="c", round_i=2, scene_profile="matched_id",
        bank=changed, temperature=1.0, role="screen",
    )


def test_v_safe_certifies_every_full_planned_horizon_and_exits_early(monkeypatch):
    spans = []

    def certificate(segment, pedestrians, gamma):
        spans.append(len(segment) - 1)
        return True, [], {}

    monkeypatch.setattr(C.SM, "certify_moving_window", certificate)
    row = dict(
        states=np.zeros((3, 4), np.float32), steps=2,
        planned_segments=np.zeros((2, C.H + 1, 2), np.float32),
        ped_xy=np.ones((2, 1, 2), np.float32) * 5,
        ped_vel=np.zeros((2, 1, 2), np.float32), gamma=.5, collision=False,
    )
    result = C._v_safe_worker(row)
    assert result == dict(v_safe=True, verifier_errors=0, windows=2)
    assert spans == [C.H, C.H]

    calls = 0

    def reject_first(segment, pedestrians, gamma):
        nonlocal calls
        calls += 1
        return False, [], {}

    monkeypatch.setattr(C.SM, "certify_moving_window", reject_first)
    assert not C._v_safe_worker(row)["v_safe"]
    assert calls == 1

    malformed = dict(row, planned_segments=np.zeros((2, C.H, 2), np.float32))
    assert C._v_safe_worker(malformed)["verifier_errors"] == 1


class _FakePolicy(torch.nn.Module):
    d = 20
    H_pred = 10
    u_max = 2.0

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(1, 1, bias=False)
        self.batch_sizes = []

    def ctx_from(self, hp10, low, history):
        return torch.zeros((len(hp10), 1), dtype=hp10.dtype, device=hp10.device)

    def forward(self, value, time, context):
        self.batch_sizes.append(len(value))
        return torch.zeros_like(value)


def test_raw_rollout_batches_all_active_episodes_per_tick(monkeypatch):
    monkeypatch.setattr(C.SS, "make_humans", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(
        C.SS, "collect_humans",
        lambda humans: (np.array([[5.0, 0.0]], np.float32), np.zeros((1, 2), np.float32)),
    )
    monkeypatch.setattr(C.SS, "advance_humans", lambda humans, state: None)
    policy = _FakePolicy()
    noise = np.zeros((len(C.SP.GAMMAS), 2, 2, policy.d), np.float32)
    rows = C.run_batched_raw(
        policy, scene_profile="matched_id", ep0=10, M=2, base_noise=noise,
        temperature=1.0, device="cpu", T_steps=2,
    )
    assert len(rows) == 2 * len(C.SP.GAMMAS)
    assert policy.batch_sizes == [len(rows)] * (2 * C.NFE)
    assert all(row["steps"] == 2 for row in rows)
    assert all(row["status"] == "timeout" for row in rows)


def test_raw_rollout_applies_the_declared_temperature_per_gamma(monkeypatch):
    monkeypatch.setattr(C.SS, "make_humans", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(
        C.SS, "collect_humans",
        lambda humans: (np.array([[5.0, 0.0]], np.float32), np.zeros((1, 2), np.float32)),
    )
    monkeypatch.setattr(C.SS, "advance_humans", lambda humans, state: None)
    observed = []

    def integrate(policy, latents, context, nfe):
        observed.extend(latents[:, 0].detach().cpu().tolist())
        return torch.zeros((len(latents), C.H, 2), device=latents.device)

    monkeypatch.setattr(C.BE, "integrate_latents", integrate)
    policy = _FakePolicy()
    noise = np.ones((len(C.SP.GAMMAS), 1, 1, policy.d), np.float32)
    temperatures = {str(gamma): .9 + .01 * index
                    for index, gamma in enumerate(C.SP.GAMMAS)}
    C.run_batched_raw(
        policy, scene_profile="matched_id", ep0=10, M=1, base_noise=noise,
        temperature=temperatures, device="cpu", T_steps=1,
    )
    assert np.allclose(observed, [temperatures[str(gamma)] for gamma in C.SP.GAMMAS])


def test_cli_matches_sweep_orchestration_contract():
    args = C.build_parser().parse_args([
        "run", "--checkpoint-dir", "x", "--scene-profile", "double_density_velocity_ood",
        "--outdir", "y", "--rounds", "0:20", "--device", "cuda:0", "--workers", "32",
    ])
    assert args.command == "run" and C.parse_rounds(args.rounds) == list(range(21))
    assert args.tune_M == 10 and args.screen_M == 50
    confirmation = C.build_parser().parse_args([
        "confirm", "--checkpoint", "r.pt", "--round", "7", "--temperature", ".95",
        "--scene-profile", "double_density_velocity_ood", "--outdir", "z",
    ])
    assert confirmation.M == 100 and confirmation.ep0 == C.SP.FINAL_CONFIRM_EP0
    vector = C.build_parser().parse_args([
        "confirm", "--checkpoint", "r.pt", "--round", "7",
        "--temperature-by-gamma", '{"0.1":0.9,"0.2":0.95,"0.3":1,"0.4":1,"0.5":1,"0.7":1.05,"1.0":1.1}',
        "--scene-profile", "double_density_velocity_ood", "--outdir", "z",
    ])
    assert vector.temperature_by_gamma.startswith("{")
    candidate = C.build_parser().parse_args([
        "candidate", "--checkpoint", "r.pt", "--round", "7",
        "--scene-profile", "double_density_velocity_ood", "--outdir", "z",
    ])
    assert candidate.command == "candidate"
    assert candidate.tune_M == 10 and candidate.screen_M == 50
    single = C.build_parser().parse_args([
        "confirm", "--checkpoint", "r.pt", "--round", "7", "--temperature", "1",
        "--single-vector-only", "--scene-profile", "double_density_velocity_ood",
        "--outdir", "z",
    ])
    assert single.single_vector_only


class _ImmediateExecutor:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _patch_compact_evaluation(monkeypatch, calls):
    monkeypatch.setattr(C, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(C.GPS, "load_sfm_policy", lambda *args, **kwargs: (_FakePolicy(), {}))
    monkeypatch.setattr(
        C, "noise_bank",
        lambda **kwargs: (
            np.zeros((1,), np.float32),
            dict(ep0=kwargs["ep0"], M=kwargs["M"], sha256=f'{kwargs["ep0"]}-{kwargs["M"]}'),
        ),
    )

    def begin(policy, **kwargs):
        calls.append(dict(kwargs))
        temperature = C.temperature_vector(kwargs["temperature"])
        return dict(
            cell_key=f"cell-{len(calls)}", temperature_by_gamma=temperature,
            summary=_summary(),
        )

    monkeypatch.setattr(C, "_begin_cell", begin)


def test_candidate_tunes_m10_then_runs_only_selected_m50(monkeypatch, tmp_path):
    calls = []
    _patch_compact_evaluation(monkeypatch, calls)
    checkpoint = tmp_path / "round_07.pt"
    checkpoint.write_bytes(b"checkpoint")
    outdir = tmp_path / "candidate"
    args = C.build_parser().parse_args([
        "candidate", "--checkpoint", str(checkpoint), "--round", "7",
        "--scene-profile", "double_density_velocity_ood", "--outdir", str(outdir),
        "--device", "cpu", "--workers", "1",
    ])
    complete = C.candidate(args)
    assert complete["status"] == "SFM_B1_CANDIDATE_SCREEN_COMPLETE"
    assert complete["round"] == 7
    assert complete["checkpoint"] == str(checkpoint.resolve())
    assert len([call for call in calls if call["M"] == C.TUNE_M]) == len(C.TEMPERATURES)
    screens = [call for call in calls if call["M"] == C.SCREEN_M]
    assert len(screens) == 1
    assert screens[0]["role"] == "candidate_selected_temperature_screening"
    assert not any("canonical" in call["role"] for call in calls)
    records = [json.loads(line) for line in (outdir / "metrics.jsonl").read_text().splitlines()]
    assert records == [dict(
        mode="candidate_selected_temperature", round=7,
        temperature_by_gamma=C.temperature_vector(1.0), summary=_summary(),
    )]


def test_single_confirmation_runs_only_supplied_vector(monkeypatch, tmp_path):
    calls = []
    _patch_compact_evaluation(monkeypatch, calls)
    checkpoint = tmp_path / "round_07.pt"
    checkpoint.write_bytes(b"checkpoint")
    outdir = tmp_path / "confirm"
    vector = C.temperature_vector(1.0)
    vector["0.1"] = .9
    args = C.build_parser().parse_args([
        "confirm", "--checkpoint", str(checkpoint), "--round", "7",
        "--temperature-by-gamma", json.dumps(vector), "--single-vector-only",
        "--scene-profile", "double_density_velocity_ood", "--outdir", str(outdir),
        "--device", "cpu", "--workers", "1",
    ])
    complete = C.confirm(args)
    assert complete["status"] == "SFM_B1_SINGLE_CONFIRMATION_COMPLETE"
    assert complete["round"] == 7
    assert complete["temperature_by_gamma"] == vector
    assert complete["summary"] == _summary()
    assert len(calls) == 1
    assert calls[0]["role"] == "final_confirmation_single_temperature"
    assert calls[0]["temperature"] == vector
    records = [json.loads(line) for line in (outdir / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["mode"] == "single_temperature_confirmation"


def test_pooled_intervals_cluster_the_shared_episode_across_gamma():
    rows = []
    for episode, success in ((10, False), (11, True)):
        for gamma in C.SP.GAMMAS:
            rows.append(dict(
                episode=episode, gamma=gamma, success=success, collision=not success,
                timeout=False, v_safe=success, verifier_errors=0, windows=1,
                successful_clearance=(.2 if success else None),
                time_to_goal=(8.0 if success else None),
            ))
    summary = C.summarize(rows, seed=4)
    pooled = summary["pooled"]
    assert "SR_wilson95" not in pooled
    assert pooled["SR_cluster_bootstrap95"] == [0.0, 1.0]
    assert pooled["ci_method"].startswith("scenario-cluster")
    assert "SR_wilson95" in summary["per_gamma"]["0.1"]
