"""Frozen two-GPU SFM B1 study orchestration, authentication, and time bound."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm as GPS
import sfm_b1_eval as BE
import sfm_b1_rbf as BR
import sfm_hp_history as HH
import sfm_protocol as SP
import sfm_scene as SS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
INPUT = "/home/dohyun/projects/cfm_mppi/overnight_run_07_12_sfm"
DATASET = os.path.join(INPUT, "dataset_id_v01")
GPU_UUIDS = {
    "1": "GPU-50fb5dae-52a8-5843-bc81-b869586dccde",
    "3": "GPU-b5993142-760d-a6fe-9430-3d0e65203b6d",
}
GPU_SLOT_TO_INDEX = {
    "1": "1", "1a": "1", "1b": "1", "1c": "1", "1d": "1",
    "3": "3", "3a": "3", "3b": "3", "3c": "3", "3d": "3",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2)
    os.replace(temporary, path)


def git_frozen_source():
    # Child scientific jobs need the conda CUDA library path, but system Git/
    # SSH on Helios must not inherit it (otherwise libssl versions conflict).
    git_environment = os.environ.copy()
    git_environment.pop("LD_LIBRARY_PATH", None)
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True, env=git_environment,
    ).strip()
    if status:
        raise RuntimeError("source worktree must be clean before a frozen run")
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True, env=git_environment,
    ).strip()
    if not branch:
        raise RuntimeError("frozen runs require a named, pushed branch")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, env=git_environment,
    ).strip()
    remote = subprocess.check_output(
        ["git", "ls-remote", "--heads", "origin", branch], cwd=ROOT, text=True,
        env=git_environment,
    ).split()
    if not remote or remote[0] != head:
        raise RuntimeError("reviewed source HEAD has not been pushed")
    return dict(branch=branch, commit=head, remote_commit=remote[0])


def gpu_snapshot():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = []
    for line in output.splitlines():
        index, uuid, name, total, used, utilization = [value.strip() for value in line.split(",")]
        rows.append(dict(index=index, uuid=uuid, name=name, memory_total_mib=int(total),
                         memory_used_mib=int(used), utilization_percent=int(utilization)))
    by_index = {row["index"]: row for row in rows}
    for index, uuid in GPU_UUIDS.items():
        if by_index.get(index, {}).get("uuid") != uuid:
            raise RuntimeError(f"GPU {index} UUID mismatch")
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader",
    ], text=True).splitlines()
    return dict(
        requested=[by_index[index] for index in ("1", "3")],
        preexisting_processes=[line for line in processes if any(uuid in line for uuid in GPU_UUIDS.values())],
        sharing_authorized_by_user=True,
    )


def cpu_pools(cpulist_path=None):
    path = Path(cpulist_path or "/sys/devices/system/node/node1/cpulist")
    if not path.exists():
        raise RuntimeError("NUMA node 1 CPU list is unavailable")
    cpus = []
    for part in path.read_text().strip().split(","):
        if "-" in part:
            low, high = map(int, part.split("-"))
            cpus.extend(range(low, high + 1))
        else:
            cpus.append(int(part))
    if len(cpus) < 64:
        raise RuntimeError("NUMA-1 does not expose 64 CPUs for disjoint pools")
    return {
        "1": cpus[:32], "3": cpus[32:64],
        "1a": cpus[:8], "1b": cpus[8:16],
        "1c": cpus[16:24], "1d": cpus[24:32],
        "3a": cpus[32:40], "3b": cpus[40:48],
        "3c": cpus[48:56], "3d": cpus[56:64],
    }


def gpu_index_for_slot(slot):
    try:
        return GPU_SLOT_TO_INDEX[str(slot)]
    except KeyError as error:
        raise ValueError(f"unknown GPU/CPU slot {slot!r}") from error


def _compact_cpu_list(values):
    return ",".join(map(str, values))


def seed_bank_manifest(outdir, rounds=20):
    payload = dict(
        declared_before_outcomes=True, gammas=list(SP.GAMMAS),
        smoke={str(round_i): list(SP.expansion_scenarios(round_i, smoke=True)) for round_i in range(1, 3)},
        expansion={str(round_i): list(SP.expansion_scenarios(round_i)) for round_i in range(1, int(rounds) + 1)},
        screening={key: list(value) for key, value in SP.raw_bank(SP.SCREEN_EP0, 20).items()},
        smoke_evaluation={key: list(value) for key, value in SP.raw_bank(SP.SMOKE_EVAL_EP0, 10).items()},
        confirmation={key: list(value) for key, value in SP.raw_bank(SP.CONFIRM_EP0, 100).items()},
        kazuki_confirmation={key: list(value) for key, value in SP.raw_bank(SP.KAZUKI_CONFIRM_EP0, 100).items()},
        deployment_id={key: list(value) for key, value in SP.raw_bank(SP.DEPLOY_ID_EP0, 100).items()},
        deployment_requested_ood={key: list(value) for key, value in SP.raw_bank(SP.DEPLOY_OOD_EP0, 100).items()},
        deployment_density_ood={
            key: list(value) for key, value in SP.raw_bank(SP.DEPLOY_DENSITY_OOD_EP0, 100).items()
        },
        temperature_selection={
            key: list(value) for key, value in SP.raw_bank(SP.TEMPERATURE_SELECT_EP0, 10).items()
        },
        curve_screening={
            key: list(value) for key, value in SP.raw_bank(SP.CURVE_SCREEN_EP0, 50).items()
        },
        final_confirmation={
            key: list(value) for key, value in SP.raw_bank(SP.FINAL_CONFIRM_EP0, 100).items()
        },
        query_diagnostic_scenarios=list(range(SP.QUERY_DIAGNOSTIC_EP0, SP.QUERY_DIAGNOSTIC_EP0 + 3)),
        environment_contracts={
            name: SS.scene_profile(name)
            for name in ("training", "id", "legacy_velocity_ood", "requested_ood", "density_ood",
                         "double_density_velocity_ood")
        },
    )
    path = os.path.join(outdir, "seed_banks.json")
    write_json(path, payload)
    return dict(path=os.path.abspath(path), sha256=sha256_file(path), payload=payload)


def authentication_manifest(outdir, checkpoint=None, scene_profile="legacy_velocity_ood"):
    source_files = sorted(Path(HERE).glob("*.py")) + sorted((Path(HERE) / "analysis").glob("test_sfm_*.py"))
    dataset_files = [Path(DATASET) / "manifest.json"] + sorted(Path(DATASET).glob("*.pt"))
    environment = SS.scene_profile(scene_profile)
    scenes = SS.scenario_snapshot(
        [scenario for round_i in range(1, 21) for scenario in SP.expansion_scenarios(round_i)],
        n_ped=environment["n_ped"], speed_range=tuple(environment["ped_speed_range"]),
    )
    scene_path = os.path.join(outdir, "scene_snapshot.json")
    write_json(scene_path, scenes)
    payload = dict(
        source={str(path): sha256_file(path) for path in source_files},
        dataset={str(path): sha256_file(path) for path in dataset_files},
        dataset_root=os.path.abspath(DATASET), scene_snapshot=os.path.abspath(scene_path),
        scene_sha256=sha256_file(scene_path),
        environment=environment,
        checkpoint=None if checkpoint is None else dict(path=os.path.abspath(checkpoint), sha256=sha256_file(checkpoint)),
    )
    path = os.path.join(outdir, "authentication.json")
    write_json(path, payload)
    payload["manifest_sha256"] = sha256_file(path)
    return payload


def _balanced_dataset_records(max_records=1200):
    per_gamma = int(np.ceil(max_records / len(SP.GAMMAS)))
    by_gamma = {}
    for gamma in SP.GAMMAS:
        path = os.path.join(DATASET, f"sfm_windows_g{gamma}.pt")
        data = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        episodes, steps = data["episode"].long(), data["step"].long()
        hp10 = HH.build_hp10(data["grid"], episodes, steps)
        groups = {int(episode): torch.nonzero(episodes == episode, as_tuple=False).flatten().tolist()
                  for episode in torch.unique(episodes, sorted=True)}
        selected = []
        cursor = 0
        keys = list(groups)
        while len(selected) < per_gamma:
            progressed = False
            for episode in keys:
                if cursor < len(groups[episode]):
                    selected.append(groups[episode][cursor])
                    progressed = True
                    if len(selected) == per_gamma:
                        break
            if not progressed:
                break
            cursor += 1
        by_gamma[float(gamma)] = []
        for index in selected:
            by_gamma[float(gamma)].append(dict(
                gamma=float(gamma), scenario_id=int(episodes[index]), step=int(steps[index]),
                hp10=hp10[index], low=data["low5"][index], hist=data["hist"][index], controls=data["U"][index],
            ))
    records = []
    for index in range(max(len(values) for values in by_gamma.values())):
        for gamma in SP.GAMMAS:
            values = by_gamma[float(gamma)]
            if index < len(values):
                records.append(values[index])
                if len(records) == max_records:
                    return records
    return records


@torch.no_grad()
def rbf_preflight(checkpoint, output, device):
    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    records = _balanced_dataset_records(1200)
    lengthscale_records = BR.balanced_exact50(records)
    def embed(values):
        parts = []
        for start in range(0, len(values), 256):
            chunk = values[start:start + 256]
            hp10 = torch.stack([row["hp10"] for row in chunk]).to(device)
            low = torch.stack([row["low"] for row in chunk]).to(device)
            hist = torch.stack([row["hist"] for row in chunk]).to(device)
            controls = torch.stack([row["controls"] for row in chunk]).to(device)
            parts.append(BR.l2_normalize(policy.phi_s(controls, policy.ctx_from(hp10, low, hist), s=.9)))
        return torch.cat(parts)
    ell0 = BR.mean_pairwise_lengthscale(embed(lengthscale_records))
    buffer_records = records[:768]
    context_records = records[768:824]
    buffer_features = embed(buffer_records)
    hp10 = torch.stack([row["hp10"] for row in context_records]).to(device)
    low = torch.stack([row["low"] for row in context_records]).to(device)
    hist = torch.stack([row["hist"] for row in context_records]).to(device)
    generator = torch.Generator(device=device).manual_seed(93471)
    windows = BE.generate_windows(policy, hp10, low, hist, K=16, nfe=8, temp=1.0, generator=generator)
    contexts = policy.ctx_from(hp10, low, hist)
    candidate_features = BR.l2_normalize(policy.phi_s(
        windows.reshape(-1, 10, 2), contexts.repeat_interleave(16, 0), s=.9
    )).reshape(len(context_records), 16, -1)
    rows = []
    for multiplier in (.5, 1.0):
        for cap in (256, 512):
            gp = BR.RBFGP(ell0 * multiplier, 1e-2)
            gp.set_buffer(buffer_features[:cap])
            try:
                beta, ess = BR.calibrate_beta(
                    gp, [candidate_features[index] for index in range(len(candidate_features))],
                    B=4, target=.5, seed=817,
                )
                selected_sigma, all_sigma = [], []
                draw_generator = torch.Generator(device=device).manual_seed(818)
                for features in candidate_features:
                    all_sigma.extend(map(float, gp.acquisition_sigma(features).cpu()))
                    _, trace = gp.sequential_acquire(features, 4, beta, generator=draw_generator)
                    selected_sigma.extend(row["chosen_sigma"] for row in trace)
                diagnostic = BR.acquisition_diagnostics(all_sigma, selected_sigma)
                gp_diagnostic = gp.diagnostics()
                rows.append(dict(
                    ell_multiplier=multiplier, ell=ell0 * multiplier, cap=cap,
                    beta=beta, ess=ess, ess_solved=True,
                    stable_conditioning=gp_diagnostic["kernel_condition"] < 1.0e8,
                    uplift=diagnostic["uplift"], acquisition=diagnostic, gp=gp_diagnostic,
                ))
            except Exception as error:
                rows.append(dict(
                    ell_multiplier=multiplier, ell=ell0 * multiplier, cap=cap,
                    ess_solved=False, stable_conditioning=False, uplift=-float("inf"), error=str(error),
                ))
    choice = BR.choose_preflight(rows)
    payload = dict(
        status="RBF_PREFLIGHT_COMPLETE", checkpoint=os.path.abspath(checkpoint),
        checkpoint_sha256=sha256_file(checkpoint), ell0=ell0,
        lengthscale_count=50, balance="round-robin gamma and scenario", lambda_=1e-2,
        candidates=rows, **choice,
    )
    write_json(output, payload)
    return payload


def _job_environment(uuid):
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", TORCH_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTHONPATH=HERE + os.pathsep + environment.get("PYTHONPATH", ""),
        LD_LIBRARY_PATH="/home/dohyun/miniforge3/lib" + os.pathsep + environment.get("LD_LIBRARY_PATH", ""),
    )
    return environment


def run_parallel(jobs, logdir):
    pools = cpu_pools()
    processes = []
    os.makedirs(logdir, exist_ok=True)
    for gpu_slot, command, name in jobs:
        gpu_index = gpu_index_for_slot(gpu_slot)
        log_path = os.path.join(logdir, name + ".log")
        stream = open(log_path, "w")
        wrapped = ["taskset", "-c", _compact_cpu_list(pools[gpu_slot]), *command]
        process = subprocess.Popen(
            wrapped, cwd=ROOT, env=_job_environment(GPU_UUIDS[gpu_index]),
            stdout=stream, stderr=subprocess.STDOUT, text=True,
        )
        processes.append((process, stream, log_path, name))
    failures = []
    for process, stream, log_path, name in processes:
        code = process.wait()
        stream.close()
        if code:
            failures.append((name, code, log_path))
    if failures:
        raise RuntimeError(f"parallel jobs failed: {failures}")
    return [item[2] for item in processes]


def full_sweep_forecast(maximum_mean_round_seconds):
    """Return JSON-native timing values for the bounded-run decision."""
    maximum_round = float(maximum_mean_round_seconds)
    forecast = float(2 * 20 * maximum_round + 3600.0)
    return maximum_round, forecast, bool(forecast <= 6 * 3600)


def smoke(checkpoint, preflight, outdir, scene_profile):
    frozen = git_frozen_source()
    gpu = gpu_snapshot()
    seeds = seed_bank_manifest(outdir)
    auth = authentication_manifest(outdir, checkpoint, scene_profile)
    selected = preflight["selected"]
    ell, cap = float(selected["ell"]), int(selected["cap"])
    python = sys.executable
    jobs = []
    for gpu_index, arm in (("1", "A"), ("3", "B")):
        arm_dir = os.path.join(outdir, f"arm_{arm}")
        jobs.append((gpu_index, [
            python, os.path.join(HERE, "sfm_b1_expand.py"), "--checkpoint", checkpoint,
            "--outdir", arm_dir, "--arm", arm, "--ell", str(ell), "--cap", str(cap),
            "--rounds", "2", "--smoke", "--device", "cuda:0", "--verifier-workers", "32",
            "--scene-profile", scene_profile,
        ], f"smoke_{arm}"))
    started = time.perf_counter()
    logs = run_parallel(jobs, os.path.join(outdir, "logs"))
    smoke_wall = time.perf_counter() - started
    evaluation_started = time.perf_counter()
    raw_reports = {}
    for round_i in (0, 2):
        eval_jobs = []
        for gpu_index, arm in (("1", "A"), ("3", "B")):
            checkpoint_path = os.path.join(outdir, f"arm_{arm}", f"round_{round_i:02d}.pt")
            report_path = os.path.join(outdir, f"arm_{arm}", f"raw_r{round_i}_M10.json")
            eval_jobs.append((gpu_index, [
                python, os.path.join(HERE, "sfm_b1_eval.py"), "--checkpoint", checkpoint_path,
                "--ep0", str(SP.SMOKE_EVAL_EP0), "--M", "10", "--device", "cuda:0", "--out", report_path,
                "--scene-profile", scene_profile,
            ], f"raw_{arm}_r{round_i}"))
            raw_reports[f"{arm}_r{round_i}"] = report_path
        logs.extend(run_parallel(eval_jobs, os.path.join(outdir, "logs")))
    evaluation_wall = time.perf_counter() - evaluation_started
    rendering_started = time.perf_counter()
    viz_report_path = os.path.join(outdir, "query_viz_report.json")
    subprocess.run([
        python, os.path.join(HERE, "sfm_b1_viz.py"),
        "--trace", os.path.join(outdir, "arm_A", "query_trace_r01.pt"),
        "--mp4", os.path.join(outdir, "query_certificate_smoke.mp4"),
        "--panels", os.path.join(outdir, "queried_candidates_smoke.png"),
        "--report", viz_report_path,
    ], cwd=ROOT, env=_job_environment(GPU_UUIDS["1"]), check=True)
    rendering_wall = time.perf_counter() - rendering_started
    arm_reports = {}
    maximum_round = 0.0
    for arm in ("A", "B"):
        manifest = json.load(open(os.path.join(outdir, f"arm_{arm}", "method_manifest.json")))
        arm_reports[arm] = manifest
        maximum_round = max(maximum_round, float(np.mean(
            [row["wall_seconds"] for row in manifest["history"]]
        )))
        if manifest["encoder_sha_before"] != manifest["encoder_sha_after"]:
            raise RuntimeError(f"arm {arm} encoder changed")
        for row in manifest["history"]:
            replay = row["replay"]
            visited = replay.get("visited", replay.get("positive_visited", []))
            eligible = replay.get("eligible", replay.get("positive_eligible", 0))
            if len(visited) != eligible or len({tuple(value) for value in visited}) != eligible:
                raise RuntimeError(f"arm {arm} replay coverage mismatch")
    # Two waves, twenty rounds each, plus the mandated one-hour final reserve.
    maximum_round, forecast, full_sweep_authorized = full_sweep_forecast(maximum_round)
    report = dict(
        status="SMOKE_COMPLETE", source=frozen, gpu=gpu, shared_gpu_override=True,
        seed_banks={key: value for key, value in seeds.items() if key != "payload"},
        authentication=auth, checkpoint=os.path.abspath(checkpoint), preflight=preflight,
        logs=logs, smoke_wall_seconds=smoke_wall, maximum_mean_round_seconds=maximum_round,
        evaluation_wall_seconds=evaluation_wall, rendering_wall_seconds=rendering_wall,
        raw_r0_r2_M10={key: dict(path=os.path.abspath(path), sha256=sha256_file(path),
                                  summary=json.load(open(path))["summary"])
                        for key, path in raw_reports.items()},
        query_visualization=dict(path=os.path.abspath(viz_report_path), sha256=sha256_file(viz_report_path)),
        full_four_arm_forecast_seconds=forecast, final_reserve_seconds=3600,
        full_sweep_authorized=full_sweep_authorized, arms=arm_reports,
    )
    write_json(os.path.join(outdir, "smoke_report.json"), report)
    return report


def _run_queue(jobs, logdir):
    """Run a job queue two at a time, one job on each requested GPU."""
    logs = []
    for start in range(0, len(jobs), 2):
        pair = []
        for offset, job in enumerate(jobs[start:start + 2]):
            pair.append(("1" if offset == 0 else "3", job[0], job[1]))
        logs.extend(run_parallel(pair, logdir))
    return logs


def full_sweep(checkpoint, preflight, outdir, scene_profile):
    """Two waves A/B then C/D, followed by frozen-bank screening and confirmation."""
    selected_rbf = preflight["selected"]
    ell, cap = float(selected_rbf["ell"]), int(selected_rbf["cap"])
    python = sys.executable
    logs = []
    full_dir = os.path.join(outdir, "full_sweep")
    for wave in (("A", "B"), ("C", "D")):
        jobs = []
        for gpu_index, arm in zip(("1", "3"), wave):
            jobs.append((gpu_index, [
                python, os.path.join(HERE, "sfm_b1_expand.py"), "--checkpoint", checkpoint,
                "--outdir", os.path.join(full_dir, f"arm_{arm}"), "--arm", arm,
                "--ell", str(ell), "--cap", str(cap), "--rounds", "20",
                "--device", "cuda:0", "--verifier-workers", "32",
                "--scene-profile", scene_profile,
            ], f"full_{arm}"))
        logs.extend(run_parallel(jobs, os.path.join(full_dir, "logs")))
    # Fixed disjoint raw M20/gamma screening at only r0/r5/r10/r15/r20.
    screen_jobs = []
    screen_paths = {}
    for arm in ("A", "B", "C", "D"):
        for round_i in (0, 5, 10, 15, 20):
            output = os.path.join(full_dir, f"arm_{arm}", f"screen_r{round_i:02d}_M20.json")
            screen_paths[(arm, round_i)] = output
            screen_jobs.append(([
                python, os.path.join(HERE, "sfm_b1_eval.py"),
                "--checkpoint", os.path.join(full_dir, f"arm_{arm}", f"round_{round_i:02d}.pt"),
                "--ep0", str(SP.SCREEN_EP0), "--M", "20", "--device", "cuda:0", "--out", output,
                "--scene-profile", scene_profile,
            ], f"screen_{arm}_r{round_i}"))
    logs.extend(_run_queue(screen_jobs, os.path.join(full_dir, "logs")))
    selections = {}
    sweep_rows = []
    for arm in ("A", "B", "C", "D"):
        candidates = []
        for round_i in (0, 5, 10, 15, 20):
            payload = json.load(open(screen_paths[(arm, round_i)]))
            key = BE.selection_key(payload["summary"])
            candidates.append((key, round_i, payload))
            sweep_rows.append(dict(
                arm=arm, round=round_i, pooled_SR=payload["summary"]["pooled"]["SR"],
                pooled_CR=payload["summary"]["pooled"]["CR"],
                worst_gamma_SR=min(row["SR"] for row in payload["summary"]["per_gamma"].values()),
                worst_gamma_CR=max(row["CR"] for row in payload["summary"]["per_gamma"].values()),
                selected=False,
            ))
        _, selected_round, selected_payload = min(candidates, key=lambda value: value[0])
        selections[arm] = dict(round=selected_round, screening=selected_payload,
                               key=list(BE.selection_key(selected_payload["summary"])))
        next(row for row in sweep_rows if row["arm"] == arm and row["round"] == selected_round)["selected"] = True
    table_path = os.path.join(full_dir, "sweep_table.csv")
    with open(table_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sweep_rows[0]))
        writer.writeheader(); writer.writerows(sweep_rows)
    # New disjoint M100/gamma bank: every selected checkpoint, its own r0, and one default Kazuki comparator.
    confirm_jobs = []
    confirm_paths = {}
    for arm in ("A", "B", "C", "D"):
        for label, round_i in (("selected", selections[arm]["round"]), ("r0", 0)):
            output = os.path.join(full_dir, f"confirm_{arm}_{label}_M100.json")
            confirm_paths[f"{arm}_{label}"] = output
            confirm_jobs.append(([
                python, os.path.join(HERE, "sfm_b1_eval.py"),
                "--checkpoint", os.path.join(full_dir, f"arm_{arm}", f"round_{round_i:02d}.pt"),
                "--ep0", str(SP.CONFIRM_EP0), "--M", "100", "--device", "cuda:0", "--out", output,
                "--scene-profile", scene_profile,
            ], f"confirm_{arm}_{label}"))
    kazuki_output = os.path.join(full_dir, "confirm_default_kazuki_M100.json")
    confirm_jobs.append(([
        python, os.path.join(HERE, "sfm_b1_sweep.py"), "kazuki-eval",
        "--checkpoint", checkpoint, "--ep0", str(SP.CONFIRM_EP0), "--M", "100",
        "--device", "cuda:0", "--out", kazuki_output, "--scene-profile", scene_profile,
    ], "confirm_kazuki"))
    logs.extend(_run_queue(confirm_jobs, os.path.join(full_dir, "logs")))
    confirmation = {key: json.load(open(path)) for key, path in confirm_paths.items()}
    confirmation["default_kazuki_generate_refine"] = json.load(open(kazuki_output))
    comparison = dict(
        status="RAW_COMPARISON_COMPLETE", selections=selections, confirmation=confirmation,
        wilson="SR/CR Wilson 95%", bootstrap="successful clearance/time and unconditional clearance bootstrap 95%",
        empirical_target_note="CR<5% is an empirical target, not a proof under real SFM dynamics",
    )
    write_json(os.path.join(full_dir, "raw_comparison_report.json"), comparison)
    overall_arm = min(
        selections,
        key=lambda arm: BE.selection_key(selections[arm]["screening"]["summary"]),
    )
    overall_round = int(selections[overall_arm]["round"])
    selected_checkpoint = os.path.join(full_dir, f"arm_{overall_arm}", f"round_{overall_round:02d}.pt")
    viz_environment = _job_environment(GPU_UUIDS["1"])
    subprocess.run([
        python, os.path.join(HERE, "sfm_b1_viz.py"),
        "--trace", os.path.join(full_dir, f"arm_{overall_arm}", f"query_trace_r{max(1, overall_round):02d}.pt"),
        "--mp4", os.path.join(full_dir, "query_certificate.mp4"),
        "--panels", os.path.join(full_dir, "queried_candidates_zoom.png"),
        "--report", os.path.join(full_dir, "query_visualization_report.json"),
    ], cwd=ROOT, env=viz_environment, check=True)
    subprocess.run([
        python, os.path.join(HERE, "sfm_b1_viz.py"),
        "--r0", checkpoint, "--selected", selected_checkpoint,
        "--scene-profile", scene_profile,
        "--gallery", os.path.join(full_dir, "selected_raw_gallery.png"),
        "--mp4", os.path.join(full_dir, "selected_raw_gallery.mp4"),
        "--device", "cuda:0", "--report", os.path.join(full_dir, "raw_gallery_report.json"),
    ], cwd=ROOT, env=viz_environment, check=True)
    artifact_hashes = {}
    for path in sorted(Path(full_dir).rglob("*")):
        if path.is_file() and path.name != "COMPLETE.json":
            artifact_hashes[str(path.relative_to(full_dir))] = sha256_file(path)
    complete = dict(
        status="COMPLETE", checkpoint=os.path.abspath(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
        source=git_frozen_source(), RBF=selected_rbf, selections=selections,
        sweep_table=os.path.abspath(table_path), logs=logs, artifact_sha256=artifact_hashes,
    )
    write_json(os.path.join(full_dir, "COMPLETE.json"), complete)
    return complete


def kazuki_evaluate(checkpoint, ep0, M, device, output, scene_profile):
    """Separately labeled generate-refine comparator; never imported by sfm_b1_eval."""
    import sfm_kazuki as KZ
    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    config = KZ.KazukiConfig(safe_coefs=(0.3,), goal_coef=0.5).validate()
    environment = SS.scene_profile(scene_profile)
    rows = []
    for gamma in SP.GAMMAS:
        for episode in range(int(ep0), int(ep0) + int(M)):
            rollout = KZ.kazuki_sfm_deploy(
                policy, episode, gamma, cfg=config, n_ped=environment["n_ped"], T=SP.T,
                device=device, ped_speed_range=tuple(environment["ped_speed_range"]),
                sample_seed=700_000, collect_diagnostics=False,
            )
            mode = "yield"
            if len(rollout["peds"]) and len(rollout["states"]) > 1:
                count = min(len(rollout["peds"]), len(rollout["states"]))
                mode = BE.classify_candidate(rollout["states"][:count, :2], rollout["peds"][:count])
            rows.append(dict(
                episode=episode, gamma=float(gamma), success=bool(rollout["success"]),
                collision=bool(rollout["collision"]), reached=bool(rollout["reached"]),
                timeout=bool(not rollout["reached"] and not rollout["collision"]), steps=int(rollout["steps"]),
                time_to_goal=(rollout["steps"] * SS.DT if rollout["success"] else None),
                min_clearance=float(rollout["min_clear"]),
                successful_clearance=(float(rollout["min_clear"]) if rollout["success"] else None),
                mode_counts={mode: 1},
            ))
    payload = dict(
        method="default Kazuki generate-refine", safe_coef=0.3, goal_coef=0.5,
        environment=environment,
        checkpoint=os.path.abspath(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
        ep0=int(ep0), M_per_gamma=int(M), summary=BE.summarize(rows), rows=rows,
    )
    write_json(output, payload)
    return payload


def write_method_readme(outdir, smoke_report):
    forecast_hours = smoke_report["full_four_arm_forecast_seconds"] / 3600
    text = "# SFM Hp10 + B1 study\n\n"
    text += "The policy consumes newest-to-oldest Hp10 and B1 gathers 8 shared OOD scenarios across seven gammas per macro-round.\n\n"
    text += f"Frozen source: `{smoke_report['source']['commit']}`. Forecast: {forecast_hours:.2f} h including the one-hour final reserve.\n"
    text += "CR<5% is an empirical target, not a proof under real SFM dynamics.\n"
    path = os.path.join(outdir, "README.md")
    os.makedirs(outdir, exist_ok=True)
    with open(path, "w") as stream:
        stream.write(text)
    return path


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--outdir", required=True)
    manifest.add_argument("--checkpoint")
    manifest.add_argument("--scene-profile", required=True, choices=SS.SCIENTIFIC_EVAL_PROFILES)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--checkpoint", required=True)
    preflight.add_argument("--out", required=True)
    preflight.add_argument("--device", default="cuda")
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--checkpoint", required=True)
    smoke_parser.add_argument("--preflight", required=True)
    smoke_parser.add_argument("--outdir", required=True)
    smoke_parser.add_argument(
        "--scene-profile", required=True,
        choices=("legacy_velocity_ood", "requested_ood", "density_ood"),
    )
    kazuki = sub.add_parser("kazuki-eval")
    kazuki.add_argument("--checkpoint", required=True)
    kazuki.add_argument("--ep0", type=int, required=True)
    kazuki.add_argument("--M", type=int, required=True)
    kazuki.add_argument("--device", default="cuda")
    kazuki.add_argument("--out", required=True)
    kazuki.add_argument("--scene-profile", required=True, choices=SS.SCIENTIFIC_EVAL_PROFILES)
    args = parser.parse_args()
    if args.command == "manifest":
        seeds = seed_bank_manifest(args.outdir)
        auth = authentication_manifest(args.outdir, args.checkpoint, args.scene_profile)
        write_json(os.path.join(args.outdir, "MANIFEST_COMPLETE.json"), dict(
            status="COMPLETE", seeds={key: value for key, value in seeds.items() if key != "payload"},
            authentication=auth, gpu=gpu_snapshot(),
        ))
    elif args.command == "preflight":
        rbf_preflight(args.checkpoint, args.out, args.device)
    elif args.command == "smoke":
        with open(args.preflight) as stream:
            selected = json.load(stream)
        report = smoke(args.checkpoint, selected, args.outdir, args.scene_profile)
        write_method_readme(args.outdir, report)
        if report["full_sweep_authorized"]:
            full_sweep(args.checkpoint, selected, args.outdir, args.scene_profile)
        else:
            write_json(os.path.join(args.outdir, "BOUNDED_STOP.json"), dict(
                status="STOPPED_BEFORE_FULL_SWEEP", reason="forecast_exceeds_six_hours",
                forecast_seconds=report["full_four_arm_forecast_seconds"],
                limit_seconds=6 * 3600, scientific_knobs_changed=False,
            ))
    else:
        kazuki_evaluate(args.checkpoint, args.ep0, args.M, args.device, args.out, args.scene_profile)


if __name__ == "__main__":
    main()
