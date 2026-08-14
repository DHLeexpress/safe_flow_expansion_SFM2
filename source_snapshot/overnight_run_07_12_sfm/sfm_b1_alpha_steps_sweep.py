"""Fail-closed two-GPU alpha x replay-epoch sweep for max-margin B1.

Every arm records canonical temperature-one M10 development metrics at every
round.  Those compact metrics shortlist four checkpoints after all training;
a disjoint M10 bank tunes their seven-gamma temperature vectors, M50 screens
them, and an untouched M100 bank confirms the locked winner.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sfm_b1_curve_eval as CE
import sfm_b1_sweep as SW


ALPHAS = (0.0, 0.001, 0.01)
INNER_EPOCHS = (1, 4, 16)
OPTIMIZER_CHUNKS = 16
SWEEP_LR = 1.0e-4
OUTPUT_ROOT = "/data3/research1"
RUNTIME_GATE_ROUNDS = 3
SCHEDULER_SLOTS = ("1a", "3a", "1b", "3b", "1c", "3c", "1d", "3d")
ARM_WORKERS = 8


def _tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _validate_output_root(path: str) -> str:
    output = os.path.realpath(path)
    output_root = os.path.realpath(OUTPUT_ROOT)
    if os.path.commonpath((output, output_root)) != output_root or output == output_root:
        raise ValueError(f"all run artifacts must use a fresh directory under {OUTPUT_ROOT}")
    return output


@dataclass(frozen=True)
class SweepArm:
    alpha: float
    inner_epochs: int

    @property
    def name(self) -> str:
        return f"margin_alpha{_tag(self.alpha)}_inner{self.inner_epochs:03d}"


def arm_grid() -> tuple[SweepArm, ...]:
    return tuple(SweepArm(alpha, epochs) for alpha in ALPHAS for epochs in INNER_EPOCHS)


def _load_preflight(path, checkpoint, expected_sha256):
    observed_sha256 = SW.sha256_file(path)
    if observed_sha256 != str(expected_sha256):
        raise RuntimeError(
            f"RBF preflight SHA-256 mismatch: {observed_sha256} != {expected_sha256}"
        )
    with open(path) as stream:
        payload = json.load(stream)
    if payload.get("status") != "RBF_PREFLIGHT_COMPLETE":
        raise RuntimeError("RBF preflight is incomplete")
    if payload.get("checkpoint_sha256") != SW.sha256_file(checkpoint):
        raise RuntimeError("RBF preflight checkpoint does not match the sweep checkpoint")
    if int(payload.get("lengthscale_count", -1)) != 50 or float(payload.get("lambda_", -1)) != 1e-2:
        raise RuntimeError("RBF preflight is not the declared exact-50/lambda=.01 contract")
    selected = payload.get("selected", {})
    if (not selected.get("ess_solved") or not selected.get("stable_conditioning")
            or int(selected.get("cap", -1)) not in (256, 512)):
        raise RuntimeError("RBF preflight has no admissible selected configuration")
    cap512 = [
        row for row in payload.get("candidates", [])
        if int(row.get("cap", -1)) == 512
        and float(row.get("ell_multiplier", -1)) == float(selected.get("ell_multiplier", -2))
        and row.get("ess_solved") and row.get("stable_conditioning")
    ]
    if len(cap512) != 1:
        raise RuntimeError("preflight lacks one stable cap-512 row at the selected length scale")
    payload["sweep_selected"] = dict(cap512[0])
    return payload


def _slot_waves(jobs, logdir):
    logs = []
    for start in range(0, len(jobs), len(SCHEDULER_SLOTS)):
        wave = []
        for slot, (command, name) in zip(
                SCHEDULER_SLOTS, jobs[start:start + len(SCHEDULER_SLOTS)]):
            wave.append((slot, command, name))
        logs.extend(SW.run_parallel(wave, logdir))
    return logs


def _sanity_rows(arm_dir, arm, *, expected_rounds):
    with open(os.path.join(arm_dir, "method_manifest.json")) as stream:
        manifest = json.load(stream)
    if (manifest.get("status") != "ARM_COMPLETE"
            or int(manifest.get("rounds", -1)) != int(expected_rounds)):
        raise RuntimeError(f"{arm.name} arm manifest is incomplete")
    payloads = [manifest["baseline_sanity"]] + [row["sanity"] for row in manifest["history"]]
    expected_indices = list(range(int(expected_rounds) + 1))
    if [int(row["round"]) for row in payloads] != expected_indices:
        raise RuntimeError(f"{arm.name} has incomplete per-round M10 sanity records")
    temperatures = {str(gamma): 1.0 for gamma in CE.SP.GAMMAS}
    rows = []
    for payload in payloads:
        if payload is None or payload.get("temperature_by_gamma") != temperatures:
            raise RuntimeError(f"{arm.name} sanity is not canonical temperature one")
        round_i = int(payload["round"])
        rows.append(dict(
            arm=arm.name, round=round_i, summary=payload["summary"],
            temperature_by_gamma=temperatures,
            checkpoint=os.path.join(arm_dir, f"round_{round_i:02d}.pt"),
        ))
    return rows


def _development_key(row, arm):
    return (*CE.temperature_selection_key(row["summary"], row["temperature_by_gamma"]),
            int(row["round"]), abs(float(arm.alpha)), int(arm.inner_epochs), arm.name)


def _development_shortlist(outdir, arms, *, expected_rounds=20):
    rows_by_arm, best_by_arm = {}, {}
    for arm in arms:
        rows = _sanity_rows(
            os.path.join(outdir, "arms", arm.name), arm,
            expected_rounds=expected_rounds,
        )
        rows_by_arm[arm] = rows
        best_by_arm[arm] = min(rows, key=lambda row: _development_key(row, arm))
    selected = []
    for alpha in ALPHAS:
        candidates = [(arm, best_by_arm[arm]) for arm in arms if arm.alpha == alpha]
        selected.append(min(candidates, key=lambda item: _development_key(item[1], item[0])))
    selected_arms = {arm for arm, _ in selected}
    remaining = [(arm, best_by_arm[arm]) for arm in arms if arm not in selected_arms]
    if remaining:
        selected.append(min(remaining, key=lambda item: _development_key(item[1], item[0])))
    if len(selected) != 4 or len({arm for arm, _ in selected}) != 4:
        raise RuntimeError("development shortlist must contain four distinct arms")
    return rows_by_arm, selected


def _candidate_record(directory):
    with open(os.path.join(directory, "COMPLETE.json")) as stream:
        complete = json.load(stream)
    if complete.get("status") != "SFM_B1_CANDIDATE_SCREEN_COMPLETE":
        raise RuntimeError(f"candidate screen incomplete: {directory}")
    with open(os.path.join(directory, "metrics.jsonl")) as stream:
        rows = [json.loads(line) for line in stream]
    matches = [row for row in rows if row.get("mode") == "candidate_selected_temperature"]
    if len(matches) != 1:
        raise RuntimeError(f"candidate screen has {len(matches)} selected records")
    record = matches[0]
    if (int(complete.get("round", -1)) != int(record["round"])
            or complete.get("temperature_by_gamma") != record["temperature_by_gamma"]
            or complete.get("screening", {}).get("summary") != record["summary"]):
        raise RuntimeError("candidate screen metrics do not match COMPLETE.json")
    return record


def screening_key(row, arm):
    return (*CE.temperature_selection_key(row["summary"], row["temperature_by_gamma"]),
            abs(float(arm.alpha)), int(arm.inner_epochs), arm.name)


def _table_row(arm, record):
    pooled = record["summary"]["pooled"]
    return dict(
        arm=arm.name, alpha=arm.alpha, optimizer_chunks=OPTIMIZER_CHUNKS,
        inner_epochs=arm.inner_epochs, round=record["round"],
        temperature_by_gamma=json.dumps(record["temperature_by_gamma"], sort_keys=True),
        SR=pooled["SR"], CR=pooled["CR"], V_safe=pooled["V_safe"],
        successful_clearance=pooled["successful_clearance"]["mean"],
        successful_time_to_goal=pooled["successful_time_to_goal"]["mean"],
    )


def _plot_arm_comparison(outdir, arms, rows_by_arm, *, winner, winning_round):
    specs = (
        ("CR", "Collision rate"), ("V_safe", r"$V_{\mathrm{safe}}$"),
        ("successful_clearance", "Successful min. clearance [m]"),
        ("successful_time_to_goal", "Successful time-to-goal [s]"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 9), constrained_layout=True)
    colors = plt.cm.viridis_r([index / max(1, len(arms) - 1) for index in range(len(arms))])
    for arm, color in zip(arms, colors):
        rows = sorted(rows_by_arm[arm], key=lambda row: int(row["round"]))
        rounds = [int(row["round"]) for row in rows]
        label = rf"$\alpha={arm.alpha:g},\ E={arm.inner_epochs}$"
        for axis, (metric, title) in zip(axes.flat, specs):
            values = []
            for row in rows:
                pooled = row["summary"]["pooled"]
                value = pooled[metric] if metric in ("CR", "V_safe") else pooled[metric]["mean"]
                values.append(np.nan if value is None else float(value))
            axis.plot(rounds, values, color=color, lw=1.45, alpha=.86, label=label)
            if arm == winner:
                index = rounds.index(int(winning_round))
                axis.plot(rounds[index], values[index], marker="*", ms=14,
                          color="#D55E00", mec="black", mew=.7)
            axis.set(title=title, xlabel="expansion round")
            axis.grid(alpha=.25)
            if metric in ("CR", "V_safe"):
                axis.set_ylim(-.03, 1.03)
    axes[0, 0].legend(ncol=3, fontsize=8)
    figure.suptitle("Fixed raw M10/gamma bank, canonical sampling temperature 1")
    for suffix in ("png", "pdf"):
        figure.savefig(os.path.join(outdir, f"arm_comparison.{suffix}"), dpi=300,
                       bbox_inches="tight")
    plt.close(figure)


def _runtime_gate(args, *, ell, cap, python, source, preflight_sha256):
    """Benchmark eight shared-GPU arms after W=2/cap-512 become active."""
    gate_started = time.perf_counter()
    root = os.path.join(args.outdir, "runtime_gate")
    # Match the saturated scientific wave exactly; the ninth arm is a lighter
    # E=1 tail and is conservatively charged at this wave's maximum time.
    specs = (
        ("runtime_alpha0_inner16", 0.0, 16),
        ("runtime_alpha0p001_inner16", .001, 16),
        ("runtime_alpha0p01_inner16", .01, 16),
        ("runtime_alpha0_inner4", 0.0, 4),
        ("runtime_alpha0p001_inner4", .001, 4),
        ("runtime_alpha0p01_inner4", .01, 4),
        ("runtime_alpha0_inner1", 0.0, 1),
        ("runtime_alpha0p001_inner1", .001, 1),
    )
    if len(specs) != len(SCHEDULER_SLOTS):
        raise AssertionError("runtime gate must saturate every declared scheduler slot")
    def command(outdir, name, alpha, epochs):
        return [
            python, os.path.join(SW.HERE, "sfm_b1_expand.py"),
            "--checkpoint", args.checkpoint, "--outdir", outdir,
            "--custom-name", name, "--selector", "margin", "--alpha", str(alpha),
            "--optimizer-steps", str(OPTIMIZER_CHUNKS), "--inner-epochs", str(epochs),
            "--lr", str(SWEEP_LR), "--sanity-M", "10",
            "--ell", str(ell), "--cap", str(cap),
            "--rounds", str(RUNTIME_GATE_ROUNDS), "--smoke",
            "--device", "cuda:0", "--verifier-workers", str(ARM_WORKERS),
            "--scene-profile", args.scene_profile,
        ]
    directories = [os.path.join(root, name) for name, _, _ in specs]
    logs = SW.run_parallel([
        (slot, command(directory, name, alpha, epochs), f"train_{name}")
        for slot, directory, (name, alpha, epochs) in zip(
            SCHEDULER_SLOTS, directories, specs
        )
    ], os.path.join(root, "logs"))
    methods = []
    for directory in directories:
        with open(os.path.join(directory, "method_manifest.json")) as stream:
            methods.append(json.load(stream))
    train_round_seconds = max(
        float(record["wall_seconds"])
        for method in methods for record in method["history"]
    )
    stage_seconds = Counter()
    for method in methods:
        for record in method["history"]:
            for stage, seconds in record["gather"]["timers"].items():
                stage_seconds[stage] = max(float(stage_seconds[stage]), float(seconds))
    baseline_seconds = max(
        float(method["baseline_sanity"]["timers"]["total"]) for method in methods
    )
    waves = math.ceil(len(arm_grid()) / len(SCHEDULER_SLOTS))
    training_seconds = waves * (
        baseline_seconds + int(args.rounds) * train_round_seconds
    )
    # Four shortlisted candidates run concurrently: 5*M10 tune + 1*M50 screen.
    candidate_screen_seconds = 10.0 * baseline_seconds
    # Canonical and locked-temperature M100 confirmations run concurrently.
    confirmation_seconds = 10.0 * baseline_seconds
    unpadded_seconds = training_seconds + candidate_screen_seconds + confirmation_seconds
    forecast_seconds = 1.10 * unpadded_seconds
    payload = dict(
        status=("RUNTIME_GATE_PASS" if forecast_seconds <= args.max_hours * 3600
                else "RUNTIME_GATE_FAIL"),
        measured_train_round_seconds=train_round_seconds,
        measured_M10_seconds=baseline_seconds,
        measured_stage_seconds=dict(stage_seconds),
        dominant_training_stage=(
            None if not stage_seconds else max(stage_seconds, key=stage_seconds.get)
        ),
        benchmark_rounds=RUNTIME_GATE_ROUNDS,
        arm_count=len(arm_grid()), parallel_slots=list(SCHEDULER_SLOTS),
        workers_per_arm=ARM_WORKERS, parallel_waves=waves, rounds=int(args.rounds),
        forecast_components=dict(
            training=training_seconds, candidate_screening=candidate_screen_seconds,
            final_confirmation=confirmation_seconds, multiplicative_headroom=1.10,
        ),
        forecast_seconds=forecast_seconds, limit_seconds=float(args.max_hours) * 3600.0,
        source_commit=source["commit"], checkpoint_sha256=SW.sha256_file(args.checkpoint),
        preflight_sha256=str(preflight_sha256), scene_profile=args.scene_profile,
        gate_wall_seconds=time.perf_counter() - gate_started,
        logs=logs,
    )
    SW.write_json(os.path.join(root, "RUNTIME_FORECAST.json"), payload)
    if payload["status"] != "RUNTIME_GATE_PASS":
        SW.write_json(os.path.join(args.outdir, "BOUNDED_STOP.json"), dict(
            status="STOPPED_BEFORE_SCIENTIFIC_SWEEP", runtime_forecast=payload,
            scientific_knobs_changed=False,
        ))
        raise RuntimeError(
            f"runtime gate failed: forecast {forecast_seconds / 3600:.2f} h "
            f"> limit {args.max_hours:.2f} h"
        )
    return payload


def _load_runtime_forecast(path, *, source, args, preflight_sha256):
    forecast_path = os.path.realpath(path)
    output_root = os.path.realpath(OUTPUT_ROOT)
    if os.path.commonpath((forecast_path, output_root)) != output_root:
        raise RuntimeError(f"runtime forecast must be stored under {OUTPUT_ROOT}")
    with open(path) as stream:
        payload = json.load(stream)
    expected = dict(
        status="RUNTIME_GATE_PASS", source_commit=source["commit"],
        checkpoint_sha256=SW.sha256_file(args.checkpoint),
        preflight_sha256=str(preflight_sha256), scene_profile=args.scene_profile,
        workers_per_arm=ARM_WORKERS, arm_count=len(arm_grid()),
        rounds=int(args.rounds), benchmark_rounds=RUNTIME_GATE_ROUNDS,
    )
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"runtime forecast {key} mismatch: {payload.get(key)!r} != {value!r}")
    if payload.get("parallel_slots") != list(SCHEDULER_SLOTS):
        raise RuntimeError("runtime forecast was not measured with the eight-slot scheduler")
    forecast_seconds = float(payload.get("forecast_seconds", math.inf))
    current_limit = float(args.max_hours) * 3600.0
    if (not math.isfinite(forecast_seconds) or forecast_seconds > current_limit
            or float(payload.get("limit_seconds", math.nan)) != current_limit):
        raise RuntimeError("runtime forecast does not satisfy the current time limit")
    for log_path in payload.get("logs", []):
        resolved = os.path.realpath(log_path)
        if (os.path.commonpath((resolved, output_root)) != output_root
                or not os.path.isfile(resolved)):
            raise RuntimeError(
                f"runtime forecast log must exist under {OUTPUT_ROOT}: {log_path}"
            )
    return payload


def run(args):
    if (int(args.rounds), int(args.tune_M), int(args.screen_M), int(args.confirm_M)) != (
            20, 10, 50, 100):
        raise ValueError("scientific sweep requires rounds=20 and disjoint M10/M50/M100 banks")
    if not math.isfinite(float(args.max_hours)) or float(args.max_hours) <= 0.0:
        raise ValueError("max-hours must be finite and positive")
    if int(args.workers) != ARM_WORKERS:
        raise ValueError(f"eight-slot scientific sweep requires {ARM_WORKERS} verifier workers per arm")
    if os.path.exists(args.outdir):
        raise FileExistsError(f"refusing existing output root: {args.outdir}")
    _validate_output_root(args.outdir)
    source = SW.git_frozen_source()
    gpu = SW.gpu_snapshot()
    if gpu["preexisting_processes"]:
        raise RuntimeError(f"GPU 1/3 are not exclusive: {gpu['preexisting_processes']}")
    preflight = _load_preflight(
        args.preflight, args.checkpoint, args.expected_preflight_sha256,
    )
    os.makedirs(args.outdir)
    started = time.perf_counter()
    seeds = SW.seed_bank_manifest(args.outdir, rounds=args.rounds)
    authentication = SW.authentication_manifest(
        args.outdir, args.checkpoint, args.scene_profile,
    )
    selected_rbf = preflight["sweep_selected"]
    ell, cap = float(selected_rbf["ell"]), int(selected_rbf["cap"])
    if cap != 512:
        raise AssertionError("latest uncertainty memory requires cap 512")
    python = sys.executable
    arms = arm_grid()
    recipe = dict(
        status="SFM_B1_ALPHA_INNER_EPOCH_SWEEP_DECLARED", source=source,
        checkpoint=os.path.abspath(args.checkpoint),
        checkpoint_sha256=SW.sha256_file(args.checkpoint),
        scene_profile=args.scene_profile, rounds=int(args.rounds),
        fixed=dict(selector="margin", K=16, B=4, T=180, W=2, batch=128,
                   lr=SWEEP_LR, optimizer_chunks=OPTIMIZER_CHUNKS,
                   sanity_M=10, ess_target=.5, ell=ell, cap=cap,
                   gp_retained_per_round=256, gp_quantile=.75,
                   gp_gamma_quota="rotating 36/37; two-round 73/74"),
        factorial=dict(alpha=list(ALPHAS), inner_epochs=list(INNER_EPOCHS)),
        preflight_sha256=SW.sha256_file(args.preflight),
        temperature=dict(grid=list(CE.TEMPERATURES), tune_M=args.tune_M,
                         screen_M=args.screen_M, shared_across_gammas=False,
                         timing="post-expansion only"),
        evaluation=("canonical temperature-1 M10 is recorded at every round; development "
                    "metrics shortlist four checkpoints; disjoint M10 tunes temperature, "
                    "one selected-vector M50 screens, and untouched M100 confirms both the "
                    "locked vector and canonical temperature one"),
        scheduler=dict(slots=list(SCHEDULER_SLOTS), workers_per_arm=ARM_WORKERS),
        runtime_limit_hours=float(args.max_hours),
    )
    SW.write_json(os.path.join(args.outdir, "recipe.json"), recipe)
    preflight_sha256 = SW.sha256_file(args.preflight)
    if args.runtime_forecast:
        if args.runtime_gate_only:
            raise ValueError("runtime-gate-only cannot consume an existing runtime forecast")
        runtime_forecast = _load_runtime_forecast(
            args.runtime_forecast, source=source, args=args,
            preflight_sha256=preflight_sha256,
        )
    else:
        runtime_forecast = _runtime_gate(
            args, ell=ell, cap=cap, python=python, source=source,
            preflight_sha256=preflight_sha256,
        )
    if args.runtime_gate_only:
        payload = dict(
            status="SFM_B1_RUNTIME_GATE_ONLY_COMPLETE",
            source=source, gpu=gpu, authentication=authentication,
            preflight=preflight, recipe=recipe, runtime_forecast=runtime_forecast,
            wall_seconds=time.perf_counter() - started,
        )
        SW.write_json(os.path.join(args.outdir, "RUNTIME_GATE_ONLY_COMPLETE.json"), payload)
        return payload

    training_jobs = []
    # Put replay-heavy arms in the saturated wave and leave a light E=1 arm
    # for the unavoidable ninth-arm tail. This changes scheduling only.
    training_order = sorted(
        arms, key=lambda arm: (-int(arm.inner_epochs), abs(float(arm.alpha)), arm.name),
    )
    for arm in training_order:
        training_jobs.append(([
            python, os.path.join(SW.HERE, "sfm_b1_expand.py"),
            "--checkpoint", args.checkpoint, "--outdir", os.path.join(args.outdir, "arms", arm.name),
            "--custom-name", arm.name, "--selector", "margin", "--alpha", str(arm.alpha),
            "--optimizer-steps", str(OPTIMIZER_CHUNKS),
            "--inner-epochs", str(arm.inner_epochs), "--lr", str(SWEEP_LR), "--sanity-M", "10",
            "--ell", str(ell), "--cap", str(cap),
            "--rounds", str(args.rounds), "--device", "cuda:0",
            "--verifier-workers", str(ARM_WORKERS),
            "--scene-profile", args.scene_profile,
        ], f"train_{arm.name}"))
    logs = list(runtime_forecast["logs"])
    logs.extend(_slot_waves(training_jobs, os.path.join(args.outdir, "logs")))

    rows_by_arm, shortlist = _development_shortlist(
        args.outdir, arms, expected_rounds=args.rounds,
    )
    development_table = []
    for arm in arms:
        row = min(rows_by_arm[arm], key=lambda item: _development_key(item, arm))
        development_table.append(_table_row(arm, row))
    development_path = os.path.join(args.outdir, "development_m10_table.csv")
    with open(development_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(development_table[0]))
        writer.writeheader(); writer.writerows(development_table)

    evaluation_jobs = []
    for arm, row in shortlist:
        evaluation_jobs.append(([
            python, os.path.join(SW.HERE, "sfm_b1_curve_eval.py"), "candidate",
            "--checkpoint", row["checkpoint"], "--round", str(row["round"]),
            "--scene-profile", args.scene_profile,
            "--outdir", os.path.join(args.outdir, "candidate_screens", arm.name),
            "--device", "cuda:0", "--workers", str(ARM_WORKERS),
            "--tune-M", str(args.tune_M), "--screen-M", str(args.screen_M),
        ], f"candidate_{arm.name}_r{int(row['round']):02d}"))
    logs.extend(_slot_waves(evaluation_jobs, os.path.join(args.outdir, "logs")))

    candidates, table = [], []
    for arm, _ in shortlist:
        screen_dir = os.path.join(args.outdir, "candidate_screens", arm.name)
        record = _candidate_record(screen_dir)
        candidates.append((screening_key(record, arm), arm, record))
        table.append(_table_row(arm, record))
    _, winner, winning_record = min(candidates, key=lambda value: value[0])
    table_path = os.path.join(args.outdir, "screening_table.csv")
    with open(table_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)

    winning_round = int(winning_record["round"])
    winning_temperatures = dict(winning_record["temperature_by_gamma"])
    _plot_arm_comparison(
        args.outdir, arms, rows_by_arm, winner=winner, winning_round=winning_round,
    )
    confirmation_dir = os.path.join(args.outdir, "confirmation")
    checkpoint = os.path.join(args.outdir, "arms", winner.name, f"round_{winning_round:02d}.pt")
    confirmation_jobs = []
    for mode, temperatures in (
            ("canonical_temp1", {str(gamma): 1.0 for gamma in CE.SP.GAMMAS}),
            ("locked_selected", winning_temperatures)):
        confirmation_jobs.append(([
            python, os.path.join(SW.HERE, "sfm_b1_curve_eval.py"), "confirm",
            "--checkpoint", checkpoint, "--round", str(winning_round),
            "--temperature-by-gamma", json.dumps(temperatures, sort_keys=True),
            "--scene-profile", args.scene_profile,
            "--outdir", os.path.join(confirmation_dir, mode),
            "--device", "cuda:0", "--workers", str(ARM_WORKERS),
            "--M", str(args.confirm_M), "--single-vector-only",
        ], f"confirm_{mode}"))
    logs.extend(SW.run_parallel([
        (slot, command, name)
        for slot, (command, name) in zip(("1a", "3a"), confirmation_jobs)
    ], os.path.join(args.outdir, "logs")))
    confirmation = {}
    for mode in ("canonical_temp1", "locked_selected"):
        with open(os.path.join(confirmation_dir, mode, "COMPLETE.json")) as stream:
            confirmation[mode] = json.load(stream)
        if confirmation[mode].get("status") != "SFM_B1_SINGLE_CONFIRMATION_COMPLETE":
            raise RuntimeError(f"final {mode} confirmation is incomplete")
    final = dict(
        status="SFM_B1_ALPHA_INNER_EPOCH_SWEEP_COMPLETE", source=source, gpu=gpu,
        checkpoint=recipe["checkpoint"], checkpoint_sha256=recipe["checkpoint_sha256"],
        scene_profile=args.scene_profile, arms=[arm.__dict__ | {"name": arm.name} for arm in arms],
        winner=dict(arm=winner.name, alpha=winner.alpha,
                    optimizer_chunks=OPTIMIZER_CHUNKS, inner_epochs=winner.inner_epochs,
                    round=winning_round, temperature_by_gamma=winning_temperatures,
                    checkpoint=os.path.abspath(checkpoint), checkpoint_sha256=SW.sha256_file(checkpoint)),
        development_m10_table=os.path.abspath(development_path),
        shortlist=[dict(arm=arm.name, round=int(row["round"])) for arm, row in shortlist],
        screening_table=os.path.abspath(table_path), confirmation=confirmation,
        comparison_plot={
            suffix: dict(
                path=os.path.abspath(os.path.join(args.outdir, f"arm_comparison.{suffix}")),
                sha256=SW.sha256_file(os.path.join(args.outdir, f"arm_comparison.{suffix}")),
            ) for suffix in ("png", "pdf")
        },
        seed_banks={key: value for key, value in seeds.items() if key != "payload"},
        authentication=authentication, preflight=preflight, logs=logs,
        runtime_forecast=runtime_forecast,
        wall_seconds=time.perf_counter() - started,
        scientific_boundary=("fixed canonical M10 is development monitoring and shortlisting; "
                             "per-gamma temperatures are chosen jointly only on a disjoint M10 "
                             "after training; one locked-vector M50 selects the arm/round; the "
                             "declared M100 bank is untouched until paired canonical/locked final "
                             "confirmation and never changes the winner"),
    )
    SW.write_json(os.path.join(args.outdir, "SWEEP_COMPLETE.json"), final)
    return final


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--scene-profile", default="double_density_velocity_ood",
                        choices=("double_density_velocity_ood", "density_ood", "requested_ood"))
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=ARM_WORKERS)
    parser.add_argument("--tune-M", type=int, default=10)
    parser.add_argument("--screen-M", type=int, default=50)
    parser.add_argument("--confirm-M", type=int, default=100)
    parser.add_argument("--max-hours", type=float, default=6.0)
    parser.add_argument("--runtime-gate-only", action="store_true")
    parser.add_argument("--runtime-forecast")
    return parser


def main(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
