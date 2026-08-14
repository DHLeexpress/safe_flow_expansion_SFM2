"""Matched raw/Kazuki deployment and checkpoint-curve evaluation for Hp10+B1.

This module is deliberately evaluation-only: raw checkpoints are sampled at
temperature one with no verifier, acquisition tilt, or execution selector.
Kazuki remains a separately named generate--guide--refine comparator.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _paths  # noqa: F401
import grid_policy_sfm as GPS
import sfm_b1_eval as BE
import sfm_kazuki as KZ
import sfm_protocol as SP
import sfm_scene as SS


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2)
    os.replace(temporary, path)


def _compact_row(row):
    return {key: value for key, value in row.items()
            if key not in ("states", "controls", "peds", "trace", "ped_vels")}


def evaluate_raw(checkpoint, bank, *, scene_profile, device):
    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    rows, summary = BE.evaluate_policy(
        policy, bank, device=device, scene_profile=scene_profile,
    )
    return dict(
        method="raw temperature-1 generative policy",
        checkpoint=os.path.abspath(checkpoint),
        checkpoint_sha256=BE.sha256_file(checkpoint),
        raw_semantics="temp=1,NFE=8,one generated window per context,execute first action; no tilt/verifier/selector",
        summary=summary, rows=[_compact_row(row) for row in rows],
    )


def evaluate_kazuki(checkpoint, bank, *, scene_profile, device,
                    safe_coef=0.3, goal_coef=0.5, variant="default"):
    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    environment = SS.scene_profile(scene_profile)
    config = KZ.KazukiConfig(
        safe_coefs=(float(safe_coef),), goal_coef=float(goal_coef),
    ).validate()
    rows = []
    for gamma in SP.GAMMAS:
        for episode in bank[str(gamma)]:
            rollout = KZ.kazuki_sfm_deploy(
                policy, episode, gamma, cfg=config,
                n_ped=environment["n_ped"], T=SP.T, device=device,
                ped_speed_range=tuple(environment["ped_speed_range"]),
                sample_seed=700_000, collect_diagnostics=False,
            )
            rows.append(dict(
                episode=int(episode), gamma=float(gamma),
                success=bool(rollout["success"]), collision=bool(rollout["collision"]),
                reached=bool(rollout["reached"]),
                timeout=bool(not rollout["reached"] and not rollout["collision"]),
                steps=int(rollout["steps"]),
                time_to_goal=(int(rollout["steps"]) * SS.DT if rollout["success"] else None),
                min_clearance=float(rollout["min_clear"]),
                successful_clearance=(float(rollout["min_clear"]) if rollout["success"] else None),
                mode_counts={},
            ))
    return dict(
        method=f"Kazuki generate-guide-refine ({variant})",
        checkpoint=os.path.abspath(checkpoint), checkpoint_sha256=BE.sha256_file(checkpoint),
        safe_coef=float(safe_coef), goal_coef=float(goal_coef), variant=str(variant),
        refinement_cost=config.refinement_cost,
        refinement_cost_manifest=(KZ.BC.scorer_manifest()
                                  if config.refinement_cost == "b1_safemppi" else None),
        comparator_semantics=("same learned prior plus reward guidance and MPPI refinement; "
                              "all MPPI refinement stages use the frozen B1 SafeMPPI proposal cost; "
                              "not raw flow; coefficients declared before comparison"),
        summary=BE.summarize(rows), rows=rows,
    )


def _summary_row(profile, label, value):
    pooled = value["summary"]["pooled"]
    return dict(
        scene_profile=profile, method=label, n=pooled["n"],
        SR=pooled["SR"], SR_lo=pooled["SR_wilson95"][0], SR_hi=pooled["SR_wilson95"][1],
        CR=pooled["CR"], CR_lo=pooled["CR_wilson95"][0], CR_hi=pooled["CR_wilson95"][1],
        successful_clearance=pooled["successful_clearance"]["mean"],
        successful_time_to_goal=pooled["successful_time_to_goal"]["mean"],
        unconditional_min_clearance=pooled["unconditional_min_clearance"]["mean"],
    )


def _render_benchmark(payload, output_png, output_csv):
    rows = [_summary_row(payload["environment"]["scene_profile"], label, value)
            for label, value in payload["methods"].items()]
    with open(output_csv, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    labels = [row["method"] for row in rows]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    axes[0].bar(x - .18, [row["SR"] for row in rows], width=.36, label="SR", color="#0072B2")
    axes[0].bar(x + .18, [row["CR"] for row in rows], width=.36, label="CR", color="#D55E00")
    axes[0].set_ylim(0, 1); axes[0].legend(); axes[0].set_ylabel("rate")
    axes[1].bar(x, [np.nan if row["successful_clearance"] is None else row["successful_clearance"]
                    for row in rows], color="#009E73")
    axes[1].set_ylabel("mean successful min clearance [m]")
    axes[2].bar(x, [np.nan if row["successful_time_to_goal"] is None else row["successful_time_to_goal"]
                    for row in rows], color="#CC79A7")
    axes[2].set_ylabel("successful time-to-goal [s]")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=18, ha="right"); axis.grid(axis="y", alpha=.2)
    figure.suptitle(
        f"Fixed raw M={payload['M_per_gamma']}/gamma bank — {payload['environment']['scene_profile']} "
        f"(n_ped={payload['environment']['n_ped']}, speed={payload['environment']['ped_speed_range']})"
    )
    figure.savefig(output_png, dpi=180)
    plt.close(figure)


def run_benchmark(r0, selected, *, scene_profile, ep0, M, device, outdir):
    environment = SS.scene_profile(scene_profile)
    bank = SP.raw_bank(ep0, M)
    methods = {
        "Hp10 r0 raw": evaluate_raw(r0, bank, scene_profile=scene_profile, device=device),
        "selected B1 raw": evaluate_raw(selected, bank, scene_profile=scene_profile, device=device),
        "default Kazuki (.3 safe, .5 goal)": evaluate_kazuki(
            r0, bank, scene_profile=scene_profile, device=device,
        ),
        "goal-stress Kazuki (.3 safe, 1.0 goal)": evaluate_kazuki(
            r0, bank, scene_profile=scene_profile, device=device,
            safe_coef=.3, goal_coef=1.0, variant="predeclared goal-stress",
        ),
    }
    payload = dict(
        status="MATCHED_DEPLOYMENT_COMPLETE", environment=environment,
        bank={key: list(value) for key, value in bank.items()}, ep0=int(ep0), M_per_gamma=int(M),
        methods=methods,
        comparison_note="All methods use the same scenario IDs per gamma; raw rows never use acquisition or verification.",
    )
    os.makedirs(outdir, exist_ok=True)
    result = os.path.join(outdir, "metrics.json")
    _write_json(result, payload)
    _render_benchmark(payload, os.path.join(outdir, "metrics.png"), os.path.join(outdir, "metrics.csv"))
    return payload


def _render_curve(records, environment, M, output):
    rounds = np.asarray([row["round"] for row in records])
    colors = plt.cm.viridis(np.linspace(.05, .95, len(SP.GAMMAS)))
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    for gamma, color in zip(SP.GAMMAS, colors):
        key = str(gamma)
        values = [row["summary"]["per_gamma"][key] for row in records]
        axes[0, 0].plot(rounds, [value["SR"] for value in values], color=color, label=f"γ={gamma}")
        axes[0, 1].plot(rounds, [value["CR"] for value in values], color=color, label=f"γ={gamma}")
        axes[1, 0].plot(rounds, [np.nan if value["successful_clearance"]["mean"] is None
                                else value["successful_clearance"]["mean"] for value in values], color=color)
        axes[1, 1].plot(rounds, [np.nan if value["successful_time_to_goal"]["mean"] is None
                                else value["successful_time_to_goal"]["mean"] for value in values], color=color)
    axes[0, 0].set_ylabel("raw SR"); axes[0, 1].set_ylabel("raw CR")
    axes[1, 0].set_ylabel("successful min clearance [m]")
    axes[1, 1].set_ylabel("successful time-to-goal [s]")
    for axis in axes.flat:
        axis.set_xlabel("checkpoint round"); axis.grid(alpha=.2); axis.set_xticks(rounds)
    axes[0, 0].set_ylim(0, 1); axes[0, 1].set_ylim(0, 1)
    axes[0, 0].legend(ncol=2, fontsize=8)
    figure.suptitle(
        f"True raw temp=1 evaluation, fixed M={M}/γ bank — {environment['scene_profile']} "
        f"(n_ped={environment['n_ped']}, speed={environment['ped_speed_range']})"
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def curve_cell(checkpoint_dir, *, scene_profile, ep0, M, round_i, device, outdir):
    environment = SS.scene_profile(scene_profile)
    bank = SP.raw_bank(ep0, M)
    os.makedirs(outdir, exist_ok=True)
    checkpoint = os.path.join(checkpoint_dir, f"round_{int(round_i):02d}.pt")
    result_path = os.path.join(outdir, f"round_{int(round_i):02d}.json")
    expected_sha = BE.sha256_file(checkpoint)
    if os.path.exists(result_path):
        with open(result_path) as stream:
            record = json.load(stream)
        if (record.get("checkpoint_sha256") != expected_sha
                or record.get("environment") != environment
                or record.get("bank") != {key: list(value) for key, value in bank.items()}):
            raise RuntimeError(f"stale curve cell: {result_path}")
        return record
    value = evaluate_raw(checkpoint, bank, scene_profile=scene_profile, device=device)
    record = dict(
        round=int(round_i), environment=environment,
        bank={key: list(value) for key, value in bank.items()}, **value,
    )
    _write_json(result_path, record)
    return record


def aggregate_curve(checkpoint_dir, *, scene_profile, ep0, M, rounds, outdir):
    environment = SS.scene_profile(scene_profile)
    bank = {key: list(value) for key, value in SP.raw_bank(ep0, M).items()}
    records = []
    for round_i in rounds:
        checkpoint = os.path.join(checkpoint_dir, f"round_{int(round_i):02d}.pt")
        result_path = os.path.join(outdir, f"round_{int(round_i):02d}.json")
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"missing curve cell: {result_path}")
        with open(result_path) as stream:
            record = json.load(stream)
        if (record.get("round") != int(round_i)
                or record.get("checkpoint_sha256") != BE.sha256_file(checkpoint)
                or record.get("environment") != environment or record.get("bank") != bank):
            raise RuntimeError(f"invalid curve cell: {result_path}")
        records.append(record)
    with open(os.path.join(outdir, "metrics.jsonl"), "w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    _render_curve(records, environment, M, os.path.join(outdir, "raw_checkpoint_curves.png"))
    _write_json(os.path.join(outdir, "COMPLETE.json"), dict(
        status="RAW_CHECKPOINT_CURVE_COMPLETE", environment=environment,
        M_per_gamma=int(M), ep0=int(ep0), rounds=list(map(int, rounds)),
        checkpoint_dir=os.path.abspath(checkpoint_dir),
    ))


def run_curve(checkpoint_dir, *, scene_profile, ep0, M, rounds, device, outdir):
    for round_i in rounds:
        curve_cell(
            checkpoint_dir, scene_profile=scene_profile, ep0=ep0, M=M,
            round_i=round_i, device=device, outdir=outdir,
        )
    aggregate_curve(
        checkpoint_dir, scene_profile=scene_profile, ep0=ep0, M=M,
        rounds=rounds, outdir=outdir,
    )


def _rounds(value):
    if ":" in value:
        start, stop = map(int, value.split(":"))
        return list(range(start, stop + 1))
    return [int(item) for item in value.split(",")]


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--r0", required=True); benchmark.add_argument("--selected", required=True)
    benchmark.add_argument("--scene-profile", required=True, choices=SS.SCIENTIFIC_EVAL_PROFILES)
    benchmark.add_argument("--ep0", type=int, required=True); benchmark.add_argument("--M", type=int, default=100)
    benchmark.add_argument("--device", default="cuda"); benchmark.add_argument("--outdir", required=True)
    curve = sub.add_parser("curve")
    curve.add_argument("--checkpoint-dir", required=True)
    curve.add_argument("--scene-profile", required=True, choices=SS.SCIENTIFIC_EVAL_PROFILES)
    curve.add_argument("--ep0", type=int, required=True); curve.add_argument("--M", type=int, default=50)
    curve.add_argument("--rounds", default="0:20"); curve.add_argument("--device", default="cuda")
    curve.add_argument("--outdir", required=True)
    cell = sub.add_parser("curve-cell")
    cell.add_argument("--checkpoint-dir", required=True)
    cell.add_argument("--scene-profile", required=True, choices=SS.SCIENTIFIC_EVAL_PROFILES)
    cell.add_argument("--ep0", type=int, required=True); cell.add_argument("--M", type=int, default=50)
    cell.add_argument("--round", type=int, required=True); cell.add_argument("--device", default="cuda")
    cell.add_argument("--outdir", required=True)
    aggregate = sub.add_parser("curve-aggregate")
    aggregate.add_argument("--checkpoint-dir", required=True)
    aggregate.add_argument("--scene-profile", required=True, choices=SS.SCIENTIFIC_EVAL_PROFILES)
    aggregate.add_argument("--ep0", type=int, required=True); aggregate.add_argument("--M", type=int, default=50)
    aggregate.add_argument("--rounds", default="0:20"); aggregate.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    if args.command == "benchmark":
        run_benchmark(
            args.r0, args.selected, scene_profile=args.scene_profile, ep0=args.ep0,
            M=args.M, device=args.device, outdir=args.outdir,
        )
    elif args.command == "curve":
        run_curve(
            args.checkpoint_dir, scene_profile=args.scene_profile, ep0=args.ep0,
            M=args.M, rounds=_rounds(args.rounds), device=args.device, outdir=args.outdir,
        )
    elif args.command == "curve-cell":
        curve_cell(
            args.checkpoint_dir, scene_profile=args.scene_profile, ep0=args.ep0,
            M=args.M, round_i=args.round, device=args.device, outdir=args.outdir,
        )
    else:
        aggregate_curve(
            args.checkpoint_dir, scene_profile=args.scene_profile, ep0=args.ep0,
            M=args.M, rounds=_rounds(args.rounds), outdir=args.outdir,
        )


if __name__ == "__main__":
    main()
