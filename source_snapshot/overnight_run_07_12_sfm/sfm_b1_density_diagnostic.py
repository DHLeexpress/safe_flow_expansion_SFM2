"""Deterministic density-OOD case search and Arm-A query-trace collection.

This module prepares inputs for paper visualizations; it is not an unbiased
policy evaluation.  ``search`` exhaustively runs every declared scenario with
the exact three displayed controllers on the same density-only OOD cells.
``collect`` then gathers diagnostic-only max-margin B1 traces for the selected
scenario.  Neither command updates a checkpoint, a round shard, or GP state on
disk.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import itertools
import json
import os

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm as GPS
import sfm_b1_expert as EX
import sfm_b1_eval as BE
import sfm_b1_expand as BX
import sfm_b1_rbf as BR
import sfm_b1_store as BS
import sfm_b1_viz_socp as VS
import sfm_kazuki as KZ
import sfm_protocol as SP
import sfm_scene as SS


DISPLAY_GAMMAS = (0.1, 0.5, 1.0)
DEFAULT_DIAGNOSTIC_EP0 = 230_000
DEFAULT_DIAGNOSTIC_N = 24
DEFAULT_ELL = 0.24210826720721101
DEFAULT_CAP = SP.GP_CAP
DEFAULT_SAMPLE_SEED = 700_000
METHODS = ("safemppi_expert", "arm_a_r10_raw", "default_kazuki")


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = os.fspath(path) + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _save_torch(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = os.fspath(path) + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _canonical_sha256(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def density_ood_environment():
    """Density-only OOD contract: training speeds and 50 pedestrians."""
    training = SS.scene_profile("training")
    speed = list(map(float, training["ped_speed_range"]))
    if int(training["n_ped"]) != 20 or speed != [0.5, 1.0]:
        raise RuntimeError("training environment no longer matches the declared density-OOD reference")
    return dict(
        scene_profile="density_ood",
        n_ped=50,
        ped_speed_range=speed,
        role="finite_density_ood_visual_diagnostic",
        shift_from_training="pedestrian count 50 versus 20; speed range unchanged",
        training_reference=dict(n_ped=20, ped_speed_range=speed),
        goal=SS.GOAL.astype(float).tolist(), pedestrian_radius=float(SS.R_PED),
        dt=float(SS.DT), sensing_radius=float(SS.R_SENSE),
    )


def visual_environment(scene_profile):
    if scene_profile == "density_ood":
        return density_ood_environment()
    return SS.scene_profile(scene_profile)


def diagnostic_bank(ep0=DEFAULT_DIAGNOSTIC_EP0, count=DEFAULT_DIAGNOSTIC_N,
                    sample_seed=DEFAULT_SAMPLE_SEED):
    if int(count) < 1:
        raise ValueError("diagnostic bank must contain at least one scenario")
    if int(ep0) < 220_000:
        raise ValueError("diagnostic bank must remain disjoint from declared banks ending at 210000")
    payload = dict(
        purpose="finite visual-diagnostic search; not unbiased evaluation",
        scenarios=list(range(int(ep0), int(ep0) + int(count))),
        gammas=list(DISPLAY_GAMMAS), sample_seed=int(sample_seed),
        environment=density_ood_environment(),
    )
    payload["bank_sha256"] = _canonical_sha256(payload)
    return payload


def _status_row(method, scenario, gamma, rollout):
    if method == "default_kazuki":
        minimum_clearance = rollout["min_clear"]
    else:
        minimum_clearance = rollout["min_clearance"]
    return dict(
        method=str(method), scenario_id=int(scenario), gamma=float(gamma),
        success=bool(rollout["success"]), collision=bool(rollout["collision"]),
        timeout=bool(not rollout["success"] and not rollout["collision"]),
        steps=int(rollout["steps"]), min_clearance=float(minimum_clearance),
    )


def _run_search_cell(policies, scenario, gamma, *, environment, device, sample_seed):
    common = dict(
        episode=int(scenario), gamma=float(gamma), device=device, T=SP.T,
        n_ped=int(environment["n_ped"]),
        ped_speed_range=tuple(environment["ped_speed_range"]),
        sample_seed=int(sample_seed),
    )
    expert = EX.rollout(
        scenario, gamma, device=device, T=SP.T, n_ped=int(environment["n_ped"]),
        ped_speed_range=tuple(environment["ped_speed_range"]),
    )
    selected = BE.raw_rollout(policies["arm_a_r10_raw"], **common)
    kazuki = KZ.kazuki_sfm_deploy(
        policies["default_kazuki"], cfg=KZ.KazukiConfig(
            safe_coefs=(0.3,), goal_coef=0.5,
        ).validate(), collect_diagnostics=False, **common,
    )
    return [
        _status_row("safemppi_expert", scenario, gamma, expert),
        _status_row("arm_a_r10_raw", scenario, gamma, selected),
        _status_row("default_kazuki", scenario, gamma, kazuki),
    ]


def scenario_score(rows):
    """Return the predeclared tier and deterministic ordering for one scenario.

    Strict tiers prioritize the requested scientific picture without hiding
    the full finite-bank search: selected succeeds at all displayed gammas and
    both comparators fail at least once; then either comparator; then selected
    succeeds throughout; then selected strictly beats both by success count.
    The final tier is an explicitly labelled fallback.
    """
    by_method = {method: [row for row in rows if row["method"] == method] for method in METHODS}
    if any(len(values) != len(DISPLAY_GAMMAS) for values in by_method.values()):
        raise ValueError("one complete three-gamma record is required per displayed method")
    successes = {method: sum(bool(row["success"]) for row in values)
                 for method, values in by_method.items()}
    failures = {method: len(DISPLAY_GAMMAS) - successes[method] for method in METHODS}
    selected_all = successes["arm_a_r10_raw"] == len(DISPLAY_GAMMAS)
    expert_failed = failures["safemppi_expert"] > 0
    kazuki_failed = failures["default_kazuki"] > 0
    if selected_all and expert_failed and kazuki_failed:
        tier, label = 0, "selected succeeds for all gammas; both comparators fail"
    elif selected_all and (expert_failed or kazuki_failed):
        tier, label = 1, "selected succeeds for all gammas; at least one comparator fails"
    elif selected_all:
        tier, label = 2, "all three methods succeed for all gammas"
    elif successes["arm_a_r10_raw"] > max(successes["safemppi_expert"], successes["default_kazuki"]):
        tier, label = 3, "selected has strictly more successes than both comparators"
    elif successes["arm_a_r10_raw"] > min(successes["safemppi_expert"], successes["default_kazuki"]):
        tier, label = 4, "selected has more successes than at least one comparator"
    else:
        tier, label = 5, "fallback: no requested dominance pattern in this scenario"
    paired_wins = sum(
        bool(selected["success"] and not comparator["success"])
        for selected in by_method["arm_a_r10_raw"]
        for comparator in by_method["safemppi_expert"] + by_method["default_kazuki"]
        if abs(float(selected["gamma"]) - float(comparator["gamma"])) <= 1.0e-9
    )
    selected_clearance = [row["min_clearance"] for row in by_method["arm_a_r10_raw"]
                          if row["success"]]
    return dict(
        tier=int(tier), tier_label=label, successes=successes, failures=failures,
        paired_selected_wins=int(paired_wins),
        selected_mean_success_clearance=(float(np.mean(selected_clearance))
                                         if selected_clearance else None),
    )


def choose_scenario(rows, scenarios):
    scored = []
    for scenario in map(int, scenarios):
        values = [row for row in rows if int(row["scenario_id"]) == scenario]
        score = scenario_score(values)
        clearance = score["selected_mean_success_clearance"]
        clearance_order = -float(clearance) if clearance is not None else 1.0e30
        ordering = (
            score["tier"], -score["successes"]["arm_a_r10_raw"],
            -score["paired_selected_wins"],
            -(score["failures"]["safemppi_expert"] + score["failures"]["default_kazuki"]),
            clearance_order, scenario,
        )
        scored.append(dict(scenario_id=scenario, score=score, ordering=list(ordering)))
    chosen = min(scored, key=lambda row: tuple(row["ordering"]))
    return chosen, scored


def run_search(r0, selected, *, ep0, count, device, sample_seed, outdir):
    bank = diagnostic_bank(ep0, count, sample_seed)
    r0_policy, _ = GPS.load_sfm_policy(r0, device=device)
    selected_policy, _ = GPS.load_sfm_policy(selected, device=device)
    policies = dict(
        arm_a_r10_raw=selected_policy.eval(),
        default_kazuki=r0_policy,
    )
    environment = bank["environment"]
    rows = []
    for scenario in bank["scenarios"]:
        for gamma in DISPLAY_GAMMAS:
            rows.extend(_run_search_cell(
                policies, scenario, gamma, environment=environment,
                device=device, sample_seed=sample_seed,
            ))
    chosen, scores = choose_scenario(rows, bank["scenarios"])
    checkpoints = dict(
        safemppi_expert=EX.manifest(),
        arm_a_r10_raw=dict(path=os.path.abspath(selected), sha256=BE.sha256_file(selected)),
        default_kazuki=dict(path=os.path.abspath(r0), sha256=BE.sha256_file(r0)),
    )
    payload = dict(
        status="DENSITY_OOD_FINITE_DIAGNOSTIC_SEARCH_COMPLETE",
        diagnostic_only=True, unbiased_evaluation=False,
        warning="Selection uses displayed outcomes on a finite bank; never report these cells as an unbiased evaluation.",
        bank=bank, controllers=dict(
            safemppi_expert=("expert reconstructed from the dataset-manifest configuration: "
                            "2048 rollouts, MPPI temp=.1, predict_gain=.25, pedestrian velocity supplied"),
            arm_a_r10_raw="temp=1,NFE=8,one raw window,execute first action; no selector at deployment",
            default_kazuki="r0 prior, safe_coef=0.3, goal_coef=0.5, generate-guide-refine",
        ), checkpoints=checkpoints, rows=rows, scenario_scores=scores,
        selection_rule=(
            "lexicographic tier: selected succeeds at all gammas with both comparator failures; "
            "then either failure; then all selected successes; then success-count dominance; "
            "fallback. Within tier maximize selected successes, paired wins, comparator failures, "
            "selected successful clearance; final tie is lowest scenario ID."
        ),
        selected_scenario_id=int(chosen["scenario_id"]), selected_score=chosen["score"],
        selected_tier=int(chosen["score"]["tier"]),
        selected_is_fallback=bool(chosen["score"]["tier"] == 5),
    )
    payload["contract_sha256"] = _canonical_sha256(dict(
        bank=bank, checkpoints=checkpoints, controllers=payload["controllers"],
        selection_rule=payload["selection_rule"],
    ))
    os.makedirs(outdir, exist_ok=False)
    _write_json(os.path.join(outdir, "diagnostic_search.json"), payload)
    _write_json(os.path.join(outdir, "selected_case.json"), dict(
        status="DENSITY_OOD_DIAGNOSTIC_CASE_SELECTED",
        diagnostic_only=True, unbiased_evaluation=False,
        scenario_id=payload["selected_scenario_id"], tier=payload["selected_tier"],
        tier_label=payload["selected_score"]["tier_label"],
        fallback=payload["selected_is_fallback"], bank_sha256=bank["bank_sha256"],
        search_contract_sha256=payload["contract_sha256"],
    ))
    return payload


def choose_shared_method_step(methods):
    """Choose one deterministic snapshot step shared by all 3x3 cells."""
    indices = []
    for method in METHODS:
        for gamma in DISPLAY_GAMMAS:
            trace = methods[method][str(gamma)].get("trace") or []
            index = {int(row["step"]): row for row in trace}
            indices.append((method, float(gamma), index))
    common = None
    for _, _, index in indices:
        steps = set(index)
        common = steps if common is None else common & steps
    if not common or 0 not in common:
        raise RuntimeError("the 3x3 method traces do not share the required initial step")
    scores = []
    for step in sorted(common):
        distances = []
        for _, _, index in indices:
            row = index[step]
            robot = np.asarray(row["state"], np.float64)[:2]
            peds = np.asarray(row["ped_xy"], np.float64)
            distances.append(float(np.linalg.norm(peds - robot[None], axis=1).min()))
        scores.append(dict(
            step=int(step), mean_min_robot_ped_center_distance=float(np.mean(distances)),
            cell_min_distances=distances,
        ))
    chosen = min(scores, key=lambda row: (
        row["mean_min_robot_ped_center_distance"], row["step"],
    ))
    return dict(
        step=int(chosen["step"]),
        mean_min_robot_ped_center_distance=chosen["mean_min_robot_ped_center_distance"],
        fallback_to_t0=bool(len(common) == 1), common_steps=list(map(int, sorted(common))),
        score_table=scores,
        rule=(
            "among steps present in all 3 methods x 3 gammas, minimize mean per-cell "
            "robot-pedestrian center distance; ties choose the earliest step; if only t=0 "
            "is shared, use t=0 and mark fallback"
        ),
    )


def validate_method_rerun(search_payload, scenario, rows):
    if int(search_payload["selected_scenario_id"]) != int(scenario):
        raise RuntimeError("method trace scenario differs from the finite-search selection")
    expected = {
        (row["method"], round(float(row["gamma"]), 8)): row
        for row in search_payload["rows"] if int(row["scenario_id"]) == int(scenario)
    }
    observed = {(row["method"], round(float(row["gamma"]), 8)): row for row in rows}
    if set(expected) != set(observed):
        raise RuntimeError("method trace rerun does not contain the selected search cells")
    for key in sorted(expected):
        for field in ("success", "collision", "timeout", "steps"):
            if expected[key][field] != observed[key][field]:
                raise RuntimeError(f"method trace rerun changed {key} field {field}")
        if not np.isclose(expected[key]["min_clearance"], observed[key]["min_clearance"], atol=1e-8):
            raise RuntimeError(f"method trace rerun changed {key} minimum clearance")
    return True


def run_methods(r0, selected, *, scenario, device, sample_seed, outdir,
                search_payload=None, scene_profile="density_ood",
                kazuki_safe_coef=.3, kazuki_goal_coef=.5):
    """Collect the complete trace bundle consumed by the 3x3 renderer."""
    if search_payload is not None:
        expected_r0 = search_payload["checkpoints"]["default_kazuki"]["sha256"]
        expected_selected = search_payload["checkpoints"]["arm_a_r10_raw"]["sha256"]
        if BE.sha256_file(r0) != expected_r0 or BE.sha256_file(selected) != expected_selected:
            raise RuntimeError("method trace checkpoints differ from the finite diagnostic search")
        if int(sample_seed) != int(search_payload["bank"]["sample_seed"]):
            raise RuntimeError("method trace sample seed differs from the finite diagnostic search")
    environment = visual_environment(scene_profile)
    kazuki_variant = (
        "default" if np.isclose(kazuki_safe_coef, .3) and np.isclose(kazuki_goal_coef, .5)
        else "goal_stress" if np.isclose(kazuki_safe_coef, .3) and np.isclose(kazuki_goal_coef, 1.0)
        else "declared_custom"
    )
    r0_policy, _ = GPS.load_sfm_policy(r0, device=device)
    selected_policy, _ = GPS.load_sfm_policy(selected, device=device)
    r0_policy.eval(); selected_policy.eval()
    methods = {method: {} for method in METHODS}
    rows = []
    for gamma in DISPLAY_GAMMAS:
        common = dict(
            episode=int(scenario), gamma=float(gamma), device=device, T=SP.T,
            n_ped=int(environment["n_ped"]),
            ped_speed_range=tuple(environment["ped_speed_range"]),
            sample_seed=int(sample_seed),
        )
        methods["safemppi_expert"][str(gamma)] = EX.rollout(
            scenario, gamma, device=device, T=SP.T, n_ped=int(environment["n_ped"]),
            ped_speed_range=tuple(environment["ped_speed_range"]), collect_trace=True,
        )
        methods["arm_a_r10_raw"][str(gamma)] = BE.raw_rollout(
            selected_policy, collect_trace=True, **common,
        )
        methods["default_kazuki"][str(gamma)] = KZ.kazuki_sfm_deploy(
            r0_policy, cfg=KZ.KazukiConfig(
                safe_coefs=(float(kazuki_safe_coef),), goal_coef=float(kazuki_goal_coef),
            ).validate(),
            collect_diagnostics=True, **common,
        )
        for method in METHODS:
            rows.append(_status_row(method, scenario, gamma, methods[method][str(gamma)]))
    rerun_matches_search = (None if search_payload is None else
                            validate_method_rerun(search_payload, scenario, rows))
    shared_snapshot = choose_shared_method_step(methods)
    bundle = dict(
        version=1, status="SFM_B1_DISPLAY_METHOD_TRACES_COMPLETE",
        diagnostic_only=True, scenario_id=int(scenario), gammas=list(DISPLAY_GAMMAS),
        sample_seed=int(sample_seed), environment=environment,
        kazuki_coefficients=dict(
            variant=kazuki_variant,
            safe_coef=float(kazuki_safe_coef), goal_coef=float(kazuki_goal_coef),
        ),
        shared_snapshot=shared_snapshot, runs=methods,
    )
    os.makedirs(outdir, exist_ok=False)
    bundle_path = os.path.join(outdir, "method_traces.pt")
    _save_torch(bundle_path, bundle)
    checkpoints = dict(
        safemppi_expert=EX.manifest(),
        arm_a_r10_raw=dict(path=os.path.abspath(selected), sha256=BE.sha256_file(selected)),
        default_kazuki=dict(path=os.path.abspath(r0), sha256=BE.sha256_file(r0)),
    )
    manifest = dict(
        status="SFM_B1_DISPLAY_METHOD_INPUTS_COMPLETE",
        diagnostic_only=True, unbiased_evaluation=False,
        scenario_id=int(scenario), gammas=list(DISPLAY_GAMMAS),
        sample_seed=int(sample_seed), environment=environment, checkpoints=checkpoints,
        kazuki_comparator=dict(
            variant=kazuki_variant,
            safe_coef=float(kazuki_safe_coef), goal_coef=float(kazuki_goal_coef),
        ),
        rows=rows, rerun_matches_search=rerun_matches_search,
        shared_snapshot=shared_snapshot, trace_bundle="method_traces.pt",
        trace_bundle_source_path=os.path.abspath(bundle_path),
        trace_bundle_sha256=BE.sha256_file(bundle_path),
        trace_contract=(
            "same scenario/gamma/environment; first row reconstructs the declared demonstration SafeMPPI expert; "
            f"A-r10 is raw temp=1/NFE=8; Kazuki uses r0, safe_coef={kazuki_safe_coef:g}, "
            f"goal_coef={kazuki_goal_coef:g} "
            "with collect_diagnostics=True"
        ),
        finite_search=(None if search_payload is None else dict(
            bank_sha256=search_payload["bank"]["bank_sha256"],
            contract_sha256=search_payload["contract_sha256"],
            selected_tier=search_payload["selected_tier"],
            selected_is_fallback=search_payload["selected_is_fallback"],
        )),
    )
    manifest["contract_sha256"] = _canonical_sha256({
            key: manifest[key] for key in (
                "scenario_id", "gammas", "sample_seed", "environment", "checkpoints",
                "kazuki_comparator", "shared_snapshot", "trace_bundle_sha256",
                "trace_contract", "finite_search",
            )
    })
    _write_json(os.path.join(outdir, "method_traces.json"), manifest)
    return manifest


def _pairwise_control_spread(rows):
    controls = [np.asarray(row["controls"], np.float64) for row in rows]
    if len(controls) < 2:
        return dict(D_U_mean=0.0, D_U_max=0.0, D_1_mean=0.0, pairs=0,
                    normalization="D_U=||Ui-Uj||/(2*u_max*sqrt(2H)); D_1=||ui0-uj0||/(2*u_max*sqrt(2))")
    full, first = [], []
    for left, right in itertools.combinations(controls, 2):
        if left.shape != right.shape or left.shape != (SP.H, 2):
            raise ValueError(f"snapshot controls must be [{SP.H},2]")
        full.append(float(np.linalg.norm(left - right) / (2.0 * SS.U_MAX * np.sqrt(2.0 * SP.H))))
        first.append(float(np.linalg.norm(left[0] - right[0]) / (2.0 * SS.U_MAX * np.sqrt(2.0))))
    return dict(
        D_U_mean=float(np.clip(np.mean(full), 0.0, 1.0)),
        D_U_max=float(np.clip(np.max(full), 0.0, 1.0)),
        D_1_mean=float(np.clip(np.mean(first), 0.0, 1.0)), pairs=len(full),
        normalization="D_U=||Ui-Uj||/(2*u_max*sqrt(2H)); D_1=||ui0-uj0||/(2*u_max*sqrt(2))",
    )


def snapshot_score(trace, trace_index):
    for row in trace["query_rows"]:
        result = row["result"]
        if result.get("resolved") and (
                not bool(result.get("full_h"))
                or int(result.get("terminal_step", -1)) != SP.H):
            raise ValueError("snapshot scoring requires every resolved query to be full H=10")
    positives = [row for row in trace["query_rows"]
                 if row["result"].get("resolved") and int(row["result"].get("y", 0)) == 1
                 and bool(row["result"].get("full_h"))]
    rejected = [row for row in trace["query_rows"]
                if row["result"].get("resolved") and int(row["result"].get("y", -1)) == 0]
    unresolved = max(0, len(trace["selected_ids"]) - len(trace["query_rows"]))
    executed = trace.get("executed_id") is not None
    all_four_resolved = (len(trace["selected_ids"]) == SP.B
                         and len(trace["query_rows"]) == SP.B and unresolved == 0)
    strict = bool(all_four_resolved and executed)
    if strict and len(positives) == 3 and len(rejected) == 1:
        tier, label = 0, "ideal P3/N1: three full-H positives and one verifier rejection"
    elif strict and len(positives) == 2 and len(rejected) == 2:
        tier, label = 1, "fallback P2/N2: two full-H positives and two verifier rejections"
    elif strict and len(positives) == 4 and len(rejected) == 0:
        tier, label = 2, "fallback P4/N0: four full-H positives and no verifier rejection"
    elif strict:
        tier, label = 3, "other all-resolved executed composition"
    elif executed:
        tier, label = 4, "non-strict executed fallback"
    else:
        tier, label = 5, "NVP trace fallback"
    spread = _pairwise_control_spread(positives)
    return dict(
        trace_index=int(trace_index), scenario_id=int(trace["scenario_id"]),
        gamma=float(trace["gamma"]), step=int(trace["step"]), tier=tier, tier_label=label,
        selected_B=len(trace["selected_ids"]), resolved=len(trace["query_rows"]),
        full_h_positive=len(positives), verifier_rejected=len(rejected),
        unresolved=unresolved,
        all_B_resolved=bool(all_four_resolved), strict_composition_eligible=bool(strict),
        executed=bool(executed), executed_id=trace.get("executed_id"), spread=spread,
    )


def choose_snapshot(traces):
    if not traces:
        raise ValueError("no query traces to score")
    scores = [snapshot_score(trace, index) for index, trace in enumerate(traces)]
    selected = min(scores, key=lambda row: (
        row["tier"], -row["spread"]["D_U_mean"],
        -row["spread"]["D_U_max"],
        -row["spread"]["D_1_mean"], row["scenario_id"],
        min(range(len(DISPLAY_GAMMAS)), key=lambda index: abs(
            DISPLAY_GAMMAS[index] - float(row["gamma"]))),
        row["step"], row["trace_index"],
    ))
    return selected, scores


def verifier_timing(traces, gather):
    attempts = sum(len(trace["selected_ids"]) for trace in traces)
    resolved = sum(len(trace["query_rows"]) for trace in traces)
    errors = attempts - resolved
    seconds = float(gather.get("timers", {}).get("verifier", 0.0))
    return dict(
        queried_attempts=int(attempts), resolved=int(resolved), errors=int(errors),
        gather_verifier_wall_seconds=seconds,
        mean_amortized_verifier_wall_ms_per_query=(1000.0 * seconds / attempts if attempts else None),
        timing_semantics=(
            "wall time around each parallel executor.map batch, including worker/IPC synchronization; "
            "the per-query value is amortized wall time, not summed single-core solver latency"
        ),
        verifier_implementation=(
            "exact 2-D angular-interval solution of each moving-disk positive max-margin "
            "SOCP block; 16 canonical artificial sensing-boundary anchors; no theta grid"
        ),
        n_theta=None, angular_grid=False, K_artificial=VS.ARTIFICIAL_FACES,
    )


def _scenario_from_search(path):
    with open(path) as stream:
        payload = json.load(stream)
    if payload.get("status") != "DENSITY_OOD_FINITE_DIAGNOSTIC_SEARCH_COMPLETE":
        raise RuntimeError("search JSON is not a completed density diagnostic")
    return int(payload["selected_scenario_id"]), payload


def run_collect(checkpoint, recent_dir, round_i, *, scenario, ell, cap, device,
                verifier_workers, seed, outdir, search_payload=None):
    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    policy.eval()
    phi_policy = copy.deepcopy(policy).eval()
    for parameter in phi_policy.parameters():
        parameter.requires_grad_(False)
    recent = BS.RecentRounds(recent_dir, SP.W)
    recent.load_through(round_i)
    gp, gp_ids = BX.gp_from_recent(
        phi_policy, recent, ell=ell, cap=cap, lam=1.0e-2,
        phi_s=0.9, device=device, seed=int(seed) + 101,
    )
    environment = density_ood_environment()
    replicas = [BX.Replica(
        int(scenario), gamma, n_ped=environment["n_ped"],
        ped_speed_range=tuple(environment["ped_speed_range"]),
    ) for gamma in DISPLAY_GAMMAS]
    cfg = BX.ArmConfig(
        name="A", selector="margin", alpha=0.0, rounds=1,
        scene_profile="density_ood", verifier_workers=int(verifier_workers), seed=int(seed),
    ).validate()
    beta, calibrated_ess = BX._initial_beta(
        phi_policy, gp, replicas, cfg, device, int(seed) + 1009,
    )
    generator = torch.Generator(device=device).manual_seed(int(seed) + 2003)
    shard = BS.RoundShard(int(round_i) + 1)
    with ProcessPoolExecutor(max_workers=int(verifier_workers)) as executor:
        gather = BX.gather_macro_round(
            policy, phi_policy, gp, beta, replicas, cfg, shard, device, executor, generator,
            record_all_traces=True,
        )
    traces = gather.pop("traces")
    selected_snapshot, snapshot_scores = choose_snapshot(traces)
    verifier = verifier_timing(traces, gather)
    os.makedirs(outdir, exist_ok=False)
    trace_path = os.path.join(outdir, "margin_traces.pt")
    snapshot_path = os.path.join(outdir, "snapshot_trace.pt")
    _save_torch(trace_path, traces)
    _save_torch(snapshot_path, traces[selected_snapshot["trace_index"]])
    report = dict(
        status="DENSITY_OOD_MARGIN_QUERY_DIAGNOSTIC_COMPLETE", diagnostic_only=True,
        enters_D_or_Dplus=False, enters_gp=False, updates_checkpoint=False,
        scenario_id=int(scenario), gammas=list(DISPLAY_GAMMAS), environment=environment,
        selector=("max one-step nominal Hp margin among resolved full-H y=1, "
                  "nominal-Hp-admissible B queries"),
        checkpoint=dict(path=os.path.abspath(checkpoint), sha256=BE.sha256_file(checkpoint)),
        recent_dir=os.path.abspath(recent_dir), recent_through_round=int(round_i),
        gp=dict(ell=float(ell), cap=int(cap), lambda_=1.0e-2,
                buffer_ids=gp_ids, diagnostics=gp.diagnostics()),
        beta=float(beta), calibrated_ess_over_K=float(calibrated_ess),
        verifier_timing=verifier, gather=gather, trace_count=len(traces),
        margin_traces="margin_traces.pt", margin_traces_source_path=os.path.abspath(trace_path),
        snapshot_trace="snapshot_trace.pt", snapshot_trace_source_path=os.path.abspath(snapshot_path),
        selected_snapshot=selected_snapshot,
        snapshot_selection_rule=(
            "strict all-B-resolved composition tiers P3/N1, then P2/N2, then P4/N0; "
            "within tier maximize normalized D_U mean/max and D_1 mean"
        ),
        snapshot_scores=snapshot_scores,
        finite_search=(None if search_payload is None else dict(
            bank_sha256=search_payload["bank"]["bank_sha256"],
            contract_sha256=search_payload["contract_sha256"],
            selected_tier=search_payload["selected_tier"],
            selected_is_fallback=search_payload["selected_is_fallback"],
        )),
    )
    _write_json(os.path.join(outdir, "query_diagnostic.json"), report)
    return report


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    search = sub.add_parser("search")
    search.add_argument("--r0", required=True); search.add_argument("--selected", required=True)
    search.add_argument("--ep0", type=int, default=DEFAULT_DIAGNOSTIC_EP0)
    search.add_argument("--count", type=int, default=DEFAULT_DIAGNOSTIC_N)
    search.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    search.add_argument("--device", default="cuda"); search.add_argument("--outdir", required=True)
    methods = sub.add_parser("methods", aliases=["render-inputs"])
    methods.add_argument("--r0", required=True); methods.add_argument("--selected", required=True)
    method_source = methods.add_mutually_exclusive_group(required=True)
    method_source.add_argument("--search-json"); method_source.add_argument("--scenario", type=int)
    methods.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    methods.add_argument("--scene-profile", choices=SS.SCIENTIFIC_EVAL_PROFILES,
                         default="density_ood")
    methods.add_argument("--kazuki-safe-coef", type=float, default=.3)
    methods.add_argument("--kazuki-goal-coef", type=float, default=.5)
    methods.add_argument("--device", default="cuda"); methods.add_argument("--outdir", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--checkpoint", required=True); collect.add_argument("--recent-dir", required=True)
    collect.add_argument("--round", type=int, default=10)
    source = collect.add_mutually_exclusive_group(required=True)
    source.add_argument("--search-json"); source.add_argument("--scenario", type=int)
    collect.add_argument("--ell", type=float, default=DEFAULT_ELL)
    collect.add_argument("--cap", type=int, default=DEFAULT_CAP)
    collect.add_argument("--device", default="cuda"); collect.add_argument("--verifier-workers", type=int, default=32)
    collect.add_argument("--seed", type=int, default=20260720); collect.add_argument("--outdir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "search":
        run_search(
            args.r0, args.selected, ep0=args.ep0, count=args.count, device=args.device,
            sample_seed=args.sample_seed, outdir=args.outdir,
        )
        return
    if args.command in ("methods", "render-inputs"):
        if args.search_json:
            scenario, search_payload = _scenario_from_search(args.search_json)
        else:
            scenario, search_payload = args.scenario, None
        run_methods(
            args.r0, args.selected, scenario=scenario, device=args.device,
            sample_seed=args.sample_seed, outdir=args.outdir, search_payload=search_payload,
            scene_profile=args.scene_profile,
            kazuki_safe_coef=args.kazuki_safe_coef,
            kazuki_goal_coef=args.kazuki_goal_coef,
        )
        return
    if args.search_json:
        scenario, search_payload = _scenario_from_search(args.search_json)
    else:
        scenario, search_payload = args.scenario, None
    run_collect(
        args.checkpoint, args.recent_dir, args.round, scenario=scenario,
        ell=args.ell, cap=args.cap, device=args.device,
        verifier_workers=args.verifier_workers, seed=args.seed,
        outdir=args.outdir, search_payload=search_payload,
    )


if __name__ == "__main__":
    main()
