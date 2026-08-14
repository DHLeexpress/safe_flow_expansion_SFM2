"""Fail-closed two-GPU driver for the corrected ID/OOD deployment package."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import sfm_b1_eval as BE
import sfm_b1_sweep as SW
import sfm_protocol as SP


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _run_jobs(jobs, logdir):
    snapshot = SW.gpu_snapshot()
    if snapshot["preexisting_processes"]:
        raise RuntimeError(f"GPU 1/3 are not exclusive: {snapshot['preexisting_processes']}")
    return SW.run_parallel(jobs, logdir), snapshot


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2)
    os.replace(temporary, path)


def run(args):
    source = SW.git_frozen_source()
    for path in (args.r0, args.selected, args.arm_dir):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    os.makedirs(args.output_root, exist_ok=False)
    python = sys.executable
    logs = []
    # Phase 1: matched M100 deployment; one complete profile per physical GPU.
    benchmark_jobs = []
    for gpu, profile, ep0 in (("1", "id", SP.DEPLOY_ID_EP0),
                              ("3", "requested_ood", SP.DEPLOY_OOD_EP0)):
        benchmark_jobs.append((gpu, [
            python, os.path.join(HERE, "sfm_b1_benchmark.py"), "benchmark",
            "--r0", args.r0, "--selected", args.selected,
            "--scene-profile", profile, "--ep0", str(ep0), "--M", str(args.deploy_M),
            "--device", "cuda:0", "--outdir", os.path.join(args.output_root, profile, "benchmark"),
        ], f"benchmark_{profile}"))
    phase_logs, gpu = _run_jobs(benchmark_jobs, os.path.join(args.output_root, "logs")); logs += phase_logs
    # Phase 2: same raw M50/gamma bank at every A checkpoint, independently by profile.
    curve_jobs = []
    for gpu_index, profile, ep0 in (("1", "id", SP.DEPLOY_ID_EP0),
                                    ("3", "requested_ood", SP.DEPLOY_OOD_EP0)):
        curve_jobs.append((gpu_index, [
            python, os.path.join(HERE, "sfm_b1_benchmark.py"), "curve",
            "--checkpoint-dir", args.arm_dir, "--scene-profile", profile,
            "--ep0", str(ep0), "--M", str(args.curve_M), "--rounds", "0:20",
            "--device", "cuda:0", "--outdir", os.path.join(args.output_root, profile, "curve"),
        ], f"curve_{profile}"))
    phase_logs, _ = _run_jobs(curve_jobs, os.path.join(args.output_root, "logs")); logs += phase_logs
    # Phase 3: fixed-scene gallery videos with honest pedestrian radii.
    gallery_jobs = []
    for gpu_index, profile, episode in (("1", "id", SP.DEPLOY_ID_EP0),
                                        ("3", "requested_ood", SP.DEPLOY_OOD_EP0)):
        directory = os.path.join(args.output_root, profile, "gallery")
        gallery_jobs.append((gpu_index, [
            python, os.path.join(HERE, "sfm_b1_viz.py"),
            "--r0", args.r0, "--selected", args.selected, "--scene-profile", profile,
            "--gallery-episode", str(episode), "--gallery", os.path.join(directory, "gallery.png"),
            "--mp4", os.path.join(directory, "gallery.mp4"), "--device", "cuda:0",
            "--report", os.path.join(directory, "gallery.json"),
        ], f"gallery_{profile}"))
    phase_logs, _ = _run_jobs(gallery_jobs, os.path.join(args.output_root, "logs")); logs += phase_logs
    # Phase 4: OOD paired selector traces.  This is diagnostic-only and never updates the model.
    query_dir = os.path.join(args.output_root, "requested_ood", "query_diagnostic")
    query_jobs = [("3", [
        python, os.path.join(HERE, "sfm_b1_query_diagnostic.py"),
        "--checkpoint", args.selected,
        "--recent-dir", os.path.join(args.arm_dir, "round_shards"), "--round", "10",
        "--scenarios", str(SP.QUERY_DIAGNOSTIC_EP0), str(SP.QUERY_DIAGNOSTIC_EP0 + 1),
        str(SP.QUERY_DIAGNOSTIC_EP0 + 2), "--ell", str(args.ell), "--cap", str(args.cap),
        "--scene-profile", "requested_ood", "--device", "cuda:0",
        "--verifier-workers", str(args.verifier_workers), "--outdir", query_dir,
    ], "query_diagnostic")]
    phase_logs, _ = _run_jobs(query_jobs, os.path.join(args.output_root, "logs")); logs += phase_logs
    diagnostic = json.load(open(os.path.join(query_dir, "diagnostic.json")))
    cases = [(row["scenario_id"], row["step"]) for row in diagnostic["shared_interaction_steps"]]
    command = [
        python, os.path.join(HERE, "sfm_b1_viz.py"),
        "--margin-trace", os.path.join(query_dir, "margin_traces.pt"),
        "--cost-trace", os.path.join(query_dir, "safemppi_cost_traces.pt"),
        "--margin-comparison", os.path.join(query_dir, "margin_3x3.png"),
        "--cost-comparison", os.path.join(query_dir, "safemppi_cost_3x3.png"),
        "--report", os.path.join(query_dir, "selector_comparison.json"),
    ]
    for scenario, step in cases:
        command.extend(["--case", f"{scenario}:{step}"])
    subprocess.run(command, cwd=ROOT, check=True)
    artifacts = {}
    for path in sorted(Path(args.output_root).rglob("*")):
        if path.is_file() and path.name != "DELIVERY_COMPLETE.json":
            artifacts[str(path.relative_to(args.output_root))] = BE.sha256_file(path)
    payload = dict(
        status="SFM_HP10_ID_OOD_DEPLOYMENT_COMPLETE", source=source,
        r0=dict(path=os.path.abspath(args.r0), sha256=BE.sha256_file(args.r0)),
        selected=dict(path=os.path.abspath(args.selected), sha256=BE.sha256_file(args.selected)),
        arm_dir=os.path.abspath(args.arm_dir), deploy_M=int(args.deploy_M), curve_M=int(args.curve_M),
        RBF=dict(ell0=0.48421653441442203, ell=float(args.ell), cap=int(args.cap), lambda_=1.0e-2,
                 preflight_beta=0.11756989408559083, target_ESS_over_K=0.5,
                 preflight_uplift=0.06455287337303162),
        gpu=gpu, logs=logs, artifacts=artifacts,
    )
    _write_json(os.path.join(args.output_root, "DELIVERY_COMPLETE.json"), payload)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--r0", required=True); parser.add_argument("--selected", required=True)
    parser.add_argument("--arm-dir", required=True); parser.add_argument("--output-root", required=True)
    parser.add_argument("--deploy-M", type=int, default=100); parser.add_argument("--curve-M", type=int, default=50)
    parser.add_argument("--ell", type=float, default=0.24210826720721101)
    parser.add_argument("--cap", type=int, default=256); parser.add_argument("--verifier-workers", type=int, default=32)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
