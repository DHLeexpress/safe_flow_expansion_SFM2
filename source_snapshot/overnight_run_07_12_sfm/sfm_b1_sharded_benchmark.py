"""Process-isolated, fail-closed ID/double-shift-OOD M100 benchmark.

Each raw/Kazuki method--gamma cell runs in its own Python process.  This is
important because the rollout implementations seed PyTorch globally.  The
aggregate is produced only after the exact declared 4 x 7 cell set has been
authenticated.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import _paths  # noqa: F401
import sfm_b1_benchmark as BB
import sfm_b1_eval as BE
import sfm_protocol as SP
import sfm_scene as SS


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PROFILES = ("matched_id", "double_density_velocity_ood")
M_PER_GAMMA = 100
METHODS = ("r0_raw", "selected_raw", "kazuki_default", "kazuki_goal_stress")
METHOD_LABELS = {
    "r0_raw": "Hp10 r0 raw",
    "selected_raw": "selected B1 raw",
    "kazuki_default": "default Kazuki (safe=0.3, goal=0.5)",
    "kazuki_goal_stress": "goal-stress Kazuki (safe=0.3, goal=1.0)",
}
METHOD_RESULT_NAMES = {
    "r0_raw": "raw temperature-1 generative policy",
    "selected_raw": "raw temperature-1 generative policy",
    "kazuki_default": "default Kazuki generate-guide-refine",
    "kazuki_goal_stress": "predeclared goal-stress Kazuki generate-guide-refine",
}
KAZUKI_CONFIGS = {
    "kazuki_default": dict(safe_coef=0.3, goal_coef=0.5),
    "kazuki_goal_stress": dict(safe_coef=0.3, goal_coef=1.0),
}


def _write_json(path, payload):
    """Atomically publish JSON; a killed cell never leaves a complete file."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    try:
        with open(temporary, "w") as stream:
            json.dump(payload, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _source_auth(expected_commit):
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    if head != expected_commit:
        raise RuntimeError(f"source commit mismatch: {head} != {expected_commit}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
    ).strip()
    if status:
        raise RuntimeError("tracked source worktree is not clean")
    return dict(commit=head, tracked_worktree_clean=True)


def _checkpoint_auth(r0, selected, expected_r0_sha256, expected_selected_sha256):
    values = {
        "r0": dict(path=os.path.abspath(r0), sha256=BE.sha256_file(r0)),
        "selected": dict(path=os.path.abspath(selected), sha256=BE.sha256_file(selected)),
    }
    expected = {"r0": expected_r0_sha256, "selected": expected_selected_sha256}
    for key in values:
        if values[key]["sha256"] != expected[key]:
            raise RuntimeError(f"{key} checkpoint SHA-256 mismatch")
    return values


def _validate_fixed_bank(scene_profile, ep0, M):
    if scene_profile not in PROFILES:
        raise ValueError(f"scene_profile must be one of {PROFILES}")
    if int(M) != M_PER_GAMMA:
        raise ValueError(f"this benchmark is fixed to M={M_PER_GAMMA} per gamma")
    if int(ep0) != int(SP.DEPLOY_DOUBLE_SHIFT_EP0):
        raise ValueError(
            f"this benchmark is fixed to ep0={SP.DEPLOY_DOUBLE_SHIFT_EP0}, shared across profiles"
        )


def _canonical_gamma(value):
    text = str(value)
    matches = [gamma for gamma in SP.GAMMAS if text == str(gamma)]
    if len(matches) != 1:
        raise ValueError(f"gamma must be one of {[str(value) for value in SP.GAMMAS]}")
    return float(matches[0])


def _base_contract(*, r0, selected, scene_profile, ep0, M,
                   expected_source_commit, expected_r0_sha256, expected_selected_sha256,
                   outdir=None, expected_gpu_uuid=None):
    _validate_fixed_bank(scene_profile, ep0, M)
    value = dict(
        source=_source_auth(expected_source_commit),
        checkpoints=_checkpoint_auth(
            r0, selected, expected_r0_sha256, expected_selected_sha256,
        ),
        scene_profile=scene_profile,
        environment=SS.scene_profile(scene_profile),
        ep0=int(ep0),
        M_per_gamma=int(M),
    )
    if expected_gpu_uuid is not None:
        value["gpu"] = _load_gpu_provenance(
            outdir, expected_gpu_uuid=expected_gpu_uuid,
            expected_source_commit=expected_source_commit,
        )
    return value


def _cell_contract(base, method, gamma):
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    gamma = _canonical_gamma(gamma)
    used_checkpoint = "selected" if method == "selected_raw" else "r0"
    value = dict(
        **base,
        method=method,
        method_label=METHOD_LABELS[method],
        gamma=gamma,
        episodes=list(range(base["ep0"], base["ep0"] + base["M_per_gamma"])),
        used_checkpoint=used_checkpoint,
    )
    if method in KAZUKI_CONFIGS:
        value["kazuki_config"] = dict(KAZUKI_CONFIGS[method])
    return value


def _gamma_slug(gamma):
    return str(_canonical_gamma(gamma)).replace(".", "p")


def cell_filename(method, gamma):
    return f"{METHODS.index(method):02d}_{method}_gamma_{_gamma_slug(gamma)}.json"


def cell_path(outdir, method, gamma):
    return os.path.join(os.path.abspath(outdir), "cells", cell_filename(method, gamma))


def _evaluate_raw_cell(checkpoint, episodes, gamma, *, scene_profile, device):
    """Exact raw evaluator semantics, restricted to one declared gamma."""
    policy, _ = BB.GPS.load_sfm_policy(checkpoint, device=device)
    environment = SS.scene_profile(scene_profile)
    rows = [BE.raw_rollout(
        policy, episode, gamma, device=device, sample_seed=700_000,
        n_ped=environment["n_ped"],
        ped_speed_range=tuple(environment["ped_speed_range"]),
    ) for episode in episodes]
    return dict(
        method=METHOD_RESULT_NAMES["r0_raw"],
        checkpoint=os.path.abspath(checkpoint), checkpoint_sha256=BE.sha256_file(checkpoint),
        raw_semantics="temp=1,NFE=8,one generated window per context,execute first action; no tilt/verifier/selector",
        rows=[BB._compact_row(row) for row in rows],
    )


def _evaluate_kazuki_cell(checkpoint, episodes, gamma, *, method, scene_profile, device):
    """Exact declared Kazuki semantics, restricted to one gamma."""
    if method not in KAZUKI_CONFIGS:
        raise ValueError(f"unknown Kazuki method: {method}")
    policy, _ = BB.GPS.load_sfm_policy(checkpoint, device=device)
    environment = SS.scene_profile(scene_profile)
    values = KAZUKI_CONFIGS[method]
    config = BB.KZ.KazukiConfig(
        safe_coefs=(values["safe_coef"],), goal_coef=values["goal_coef"],
    ).validate()
    rows = []
    for episode in episodes:
        rollout = BB.KZ.kazuki_sfm_deploy(
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
        method=METHOD_RESULT_NAMES[method],
        checkpoint=os.path.abspath(checkpoint), checkpoint_sha256=BE.sha256_file(checkpoint),
        safe_coef=values["safe_coef"], goal_coef=values["goal_coef"],
        refinement_cost=config.refinement_cost,
        refinement_cost_manifest=(BB.KZ.BC.scorer_manifest()
                                  if config.refinement_cost == "b1_safemppi" else None),
        comparator_semantics=("learned prior plus reward guidance and MPPI refinement; "
                              "all refinement stages use the frozen B1 SafeMPPI proposal cost; not raw flow"),
        rows=rows,
    )


def _validate_cell(payload, expected_contract):
    if payload.get("status") != "SFM_B1_BENCHMARK_CELL_COMPLETE":
        raise RuntimeError("cell is not complete")
    if payload.get("contract") != expected_contract:
        raise RuntimeError("cell contract mismatch")
    result = payload.get("result", {})
    if result.get("method") != METHOD_RESULT_NAMES[expected_contract["method"]]:
        raise RuntimeError("cell result method mismatch")
    if expected_contract["method"] in KAZUKI_CONFIGS:
        expected_config = KAZUKI_CONFIGS[expected_contract["method"]]
        observed_config = {
            "safe_coef": result.get("safe_coef"),
            "goal_coef": result.get("goal_coef"),
        }
        if observed_config != expected_config:
            raise RuntimeError("cell Kazuki configuration mismatch")
        if result.get("refinement_cost") != "b1_safemppi":
            raise RuntimeError("cell Kazuki refinement cost is not B1-matched")
    checkpoint = expected_contract["checkpoints"][expected_contract["used_checkpoint"]]
    if (result.get("checkpoint") != checkpoint["path"]
            or result.get("checkpoint_sha256") != checkpoint["sha256"]):
        raise RuntimeError("cell result checkpoint mismatch")
    rows = result.get("rows", [])
    episodes = expected_contract["episodes"]
    if len(rows) != len(episodes):
        raise RuntimeError("cell row count mismatch")
    observed = [(row.get("episode"), row.get("gamma")) for row in rows]
    expected = [(episode, expected_contract["gamma"]) for episode in episodes]
    if observed != expected or len(set(observed)) != len(observed):
        raise RuntimeError("cell rows do not match the fixed episode/gamma bank")
    return payload


def run_cell(*, r0, selected, scene_profile, ep0, M, method, gamma, device, outdir,
             expected_source_commit, expected_r0_sha256, expected_selected_sha256,
             expected_gpu_uuid):
    base = _base_contract(
        r0=r0, selected=selected, scene_profile=scene_profile, ep0=ep0, M=M,
        expected_source_commit=expected_source_commit,
        expected_r0_sha256=expected_r0_sha256,
        expected_selected_sha256=expected_selected_sha256,
        outdir=outdir, expected_gpu_uuid=expected_gpu_uuid,
    )
    contract = _cell_contract(base, method, gamma)
    output = cell_path(outdir, method, gamma)
    if os.path.exists(output):
        with open(output) as stream:
            return _validate_cell(json.load(stream), contract)
    if method in KAZUKI_CONFIGS:
        result = _evaluate_kazuki_cell(
            r0, contract["episodes"], contract["gamma"],
            method=method, scene_profile=scene_profile, device=device,
        )
    else:
        checkpoint = selected if method == "selected_raw" else r0
        result = _evaluate_raw_cell(
            checkpoint, contract["episodes"], contract["gamma"],
            scene_profile=scene_profile, device=device,
        )
    payload = dict(status="SFM_B1_BENCHMARK_CELL_COMPLETE", contract=contract, result=result)
    _validate_cell(payload, contract)
    _write_json(output, payload)
    return payload


def scenario_cluster_bootstrap(rows, *, seed=9417, draws=10_000):
    """Resample episode IDs while preserving their seven paired gamma rows."""
    episodes = sorted({int(row["episode"]) for row in rows})
    by_episode = {episode: [row for row in rows if int(row["episode"]) == episode]
                  for episode in episodes}
    for episode, values in by_episode.items():
        if [float(row["gamma"]) for row in values] != list(SP.GAMMAS):
            raise RuntimeError(f"episode {episode} does not contain exactly seven sorted gamma rows")
    success = np.asarray([
        [float(row["success"]) for row in by_episode[episode]] for episode in episodes
    ])
    collision = np.asarray([
        [float(row["collision"]) for row in by_episode[episode]] for episode in episodes
    ])
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, len(episodes), size=(int(draws), len(episodes)))

    def metric(values):
        estimates = values[indices].mean(axis=(1, 2))
        return dict(
            estimate=float(values.mean()),
            interval95=list(map(float, np.quantile(estimates, [.025, .975]))),
        )

    return dict(
        unit="episode_id with all seven gamma rows preserved",
        clusters=len(episodes), rows_per_cluster=len(SP.GAMMAS),
        draws=int(draws), seed=int(seed), SR=metric(success), CR=metric(collision),
    )


def _aggregate_method(cells, method, *, bootstrap_seed):
    results = [cell["result"] for cell in cells]
    metadata = [{key: value for key, value in result.items() if key not in ("rows", "summary")}
                for result in results]
    if any(value != metadata[0] for value in metadata[1:]):
        raise RuntimeError(f"inconsistent result metadata for {method}")
    rows = [row for result in results for row in result["rows"]]
    summary = BE.summarize(rows)
    return dict(
        **metadata[0], rows=rows, summary=summary,
        pooled_scenario_cluster_bootstrap95=scenario_cluster_bootstrap(
            rows, seed=bootstrap_seed,
        ),
    )


def aggregate(*, r0, selected, scene_profile, ep0, M, outdir,
              expected_source_commit, expected_r0_sha256, expected_selected_sha256,
              expected_gpu_uuid):
    base = _base_contract(
        r0=r0, selected=selected, scene_profile=scene_profile, ep0=ep0, M=M,
        expected_source_commit=expected_source_commit,
        expected_r0_sha256=expected_r0_sha256,
        expected_selected_sha256=expected_selected_sha256,
        outdir=outdir, expected_gpu_uuid=expected_gpu_uuid,
    )
    pairs = [(method, gamma) for method in METHODS for gamma in SP.GAMMAS]
    expected_names = [cell_filename(method, gamma) for method, gamma in pairs]
    cells_dir = Path(outdir) / "cells"
    observed_names = sorted(path.name for path in cells_dir.glob("*.json")) if cells_dir.exists() else []
    if observed_names != sorted(expected_names):
        raise RuntimeError(
            f"aggregate requires exactly {len(expected_names)} declared cells; "
            f"observed={observed_names}"
        )
    loaded = {}
    for method, gamma in pairs:
        path = cells_dir / cell_filename(method, gamma)
        with open(path) as stream:
            payload = json.load(stream)
        key = (method, float(gamma))
        if key in loaded:
            raise RuntimeError(f"duplicate cell key: {key}")
        loaded[key] = _validate_cell(payload, _cell_contract(base, method, gamma))
    if len(loaded) != len(pairs):
        raise RuntimeError("cell key cardinality mismatch")
    methods = {}
    for index, method in enumerate(METHODS):
        method_cells = [loaded[(method, float(gamma))] for gamma in SP.GAMMAS]
        methods[METHOD_LABELS[method]] = _aggregate_method(
            method_cells, method, bootstrap_seed=9417 + index,
        )
    payload = dict(
        status="MATCHED_SHARDED_DEPLOYMENT_COMPLETE",
        source=base["source"], checkpoints=base["checkpoints"],
        gpu=base["gpu"],
        environment=base["environment"],
        bank={str(gamma): list(range(int(ep0), int(ep0) + int(M))) for gamma in SP.GAMMAS},
        ep0=int(ep0), M_per_gamma=int(M), methods=methods,
        cell_order=[dict(method=method, gamma=float(gamma), file=cell_filename(method, gamma))
                    for method, gamma in pairs],
        comparison_note=(
            "All methods use the same scenario IDs per gamma; raw rows never use acquisition or "
            "verification. Pooled SR/CR intervals resample scenario IDs with all seven gamma rows."
        ),
    )
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    metrics = os.path.join(outdir, "metrics.json")
    png = os.path.join(outdir, "metrics.png")
    csv = os.path.join(outdir, "metrics.csv")
    _write_json(metrics, payload)
    temporary_png, temporary_csv = png + ".tmp.png", csv + ".tmp.csv"
    BB._render_benchmark(payload, temporary_png, temporary_csv)
    os.replace(temporary_png, png)
    os.replace(temporary_csv, csv)
    cell_hashes = {name: BE.sha256_file(cells_dir / name) for name in expected_names}
    complete = dict(
        status="SFM_B1_SHARDED_BENCHMARK_COMPLETE", source=base["source"],
        gpu_provenance=dict(
            path=os.path.abspath(os.path.join(outdir, "gpu_provenance.json")),
            sha256=BE.sha256_file(os.path.join(outdir, "gpu_provenance.json")),
            **base["gpu"],
        ),
        scene_profile=scene_profile, ep0=int(ep0), M_per_gamma=int(M),
        cell_count=len(pairs), cell_hashes=cell_hashes,
        artifacts={name: BE.sha256_file(os.path.join(outdir, name))
                   for name in ("metrics.json", "metrics.png", "metrics.csv")},
    )
    _write_json(os.path.join(outdir, "COMPLETE.json"), complete)
    return payload


def _driver_environment(cuda_visible_device):
    value = str(cuda_visible_device).strip()
    if not value or "," in value:
        raise ValueError("declare exactly one CUDA-visible GPU index or UUID")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = value
    environment["PYTHONPATH"] = HERE + os.pathsep + environment.get("PYTHONPATH", "")
    return environment


def _gpu_snapshot(cuda_visible_device, expected_gpu_uuid, source):
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,driver_version",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = []
    for line in output.splitlines():
        index, uuid, name, driver = [value.strip() for value in line.split(",", 3)]
        rows.append(dict(index=index, uuid=uuid, name=name, driver_version=driver))
    declared = str(cuda_visible_device).strip()
    matches = [row for row in rows if declared in (row["index"], row["uuid"])]
    if len(matches) != 1:
        raise RuntimeError(f"declared CUDA-visible device does not resolve uniquely: {declared}")
    selected = matches[0]
    if selected["uuid"] != expected_gpu_uuid:
        raise RuntimeError(f"GPU UUID mismatch: {selected['uuid']} != {expected_gpu_uuid}")
    return dict(
        status="SFM_B1_GPU_PROVENANCE", declared_cuda_visible_device=declared,
        index=selected["index"], uuid=selected["uuid"], name=selected["name"],
        driver_version=selected["driver_version"], source=source,
    )


def _load_gpu_provenance(outdir, *, expected_gpu_uuid, expected_source_commit):
    path = os.path.join(os.path.abspath(outdir), "gpu_provenance.json")
    if not os.path.exists(path):
        raise RuntimeError("missing gpu_provenance.json")
    with open(path) as stream:
        payload = json.load(stream)
    if (payload.get("status") != "SFM_B1_GPU_PROVENANCE"
            or payload.get("uuid") != expected_gpu_uuid
            or payload.get("source", {}).get("commit") != expected_source_commit
            or payload.get("source", {}).get("tracked_worktree_clean") is not True):
        raise RuntimeError("GPU provenance contract mismatch")
    return payload


def _contract_args(args):
    return [
        "--r0", args.r0, "--selected", args.selected,
        "--scene-profile", args.scene_profile, "--ep0", str(args.ep0), "--M", str(args.M),
        "--outdir", args.outdir,
        "--expected-source-commit", args.expected_source_commit,
        "--expected-r0-sha256", args.expected_r0_sha256,
        "--expected-selected-sha256", args.expected_selected_sha256,
        "--expected-gpu-uuid", args.expected_gpu_uuid,
    ]


def _cell_command(args, method, gamma):
    return [
        sys.executable, os.path.abspath(__file__), "cell", *_contract_args(args),
        "--method", method, "--gamma", str(gamma), "--device", "cuda:0",
    ]


def run_driver(args):
    if int(args.max_processes) < 1:
        raise ValueError("max_processes must be positive")
    # Authenticate before launching any subprocess.
    _base_contract(
        r0=args.r0, selected=args.selected, scene_profile=args.scene_profile,
        ep0=args.ep0, M=args.M, expected_source_commit=args.expected_source_commit,
        expected_r0_sha256=args.expected_r0_sha256,
        expected_selected_sha256=args.expected_selected_sha256,
    )
    os.makedirs(os.path.abspath(args.outdir), exist_ok=True)
    provenance_path = os.path.join(os.path.abspath(args.outdir), "gpu_provenance.json")
    provenance = _gpu_snapshot(
        args.cuda_visible_device, args.expected_gpu_uuid,
        _source_auth(args.expected_source_commit),
    )
    if os.path.exists(provenance_path):
        with open(provenance_path) as stream:
            existing = json.load(stream)
        if existing != provenance:
            raise RuntimeError("existing GPU provenance does not match this driver")
    else:
        _write_json(provenance_path, provenance)
    environment = _driver_environment(args.cuda_visible_device)
    jobs = [(method, gamma) for method in METHODS for gamma in SP.GAMMAS]
    active = []
    logs = os.path.join(os.path.abspath(args.outdir), "logs")
    os.makedirs(logs, exist_ok=True)
    try:
        while jobs or active:
            while jobs and len(active) < int(args.max_processes):
                method, gamma = jobs.pop(0)
                log_path = os.path.join(logs, cell_filename(method, gamma).replace(".json", ".log"))
                stream = open(log_path, "w")
                process = subprocess.Popen(
                    _cell_command(args, method, gamma), cwd=ROOT, env=environment,
                    stdout=stream, stderr=subprocess.STDOUT, text=True,
                )
                active.append((process, stream, log_path))
            finished = []
            for item in active:
                code = item[0].poll()
                if code is None:
                    continue
                item[1].close()
                if code:
                    raise RuntimeError(f"benchmark cell failed ({code}): {item[2]}")
                finished.append(item)
            active = [item for item in active if item not in finished]
            if active and not finished:
                time.sleep(.1)
    except BaseException:
        for process, stream, _ in active:
            if process.poll() is None:
                process.terminate()
        for process, stream, _ in active:
            process.wait()
            stream.close()
        raise
    command = [sys.executable, os.path.abspath(__file__), "aggregate", *_contract_args(args)]
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def _add_contract_arguments(parser):
    parser.add_argument("--r0", required=True)
    parser.add_argument("--selected", required=True)
    parser.add_argument("--scene-profile", required=True, choices=PROFILES)
    parser.add_argument("--ep0", type=int, default=SP.DEPLOY_DOUBLE_SHIFT_EP0)
    parser.add_argument("--M", type=int, default=M_PER_GAMMA)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-r0-sha256", required=True)
    parser.add_argument("--expected-selected-sha256", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    cell = commands.add_parser("cell")
    _add_contract_arguments(cell)
    cell.add_argument("--method", required=True, choices=METHODS)
    cell.add_argument("--gamma", required=True, choices=tuple(map(str, SP.GAMMAS)))
    cell.add_argument("--device", default="cuda:0")
    aggregate_parser = commands.add_parser("aggregate")
    _add_contract_arguments(aggregate_parser)
    driver = commands.add_parser("driver")
    _add_contract_arguments(driver)
    driver.add_argument("--cuda-visible-device", required=True)
    driver.add_argument("--max-processes", type=int, default=1)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    values = dict(
        r0=args.r0, selected=args.selected, scene_profile=args.scene_profile,
        ep0=args.ep0, M=args.M, outdir=args.outdir,
        expected_source_commit=args.expected_source_commit,
        expected_r0_sha256=args.expected_r0_sha256,
        expected_selected_sha256=args.expected_selected_sha256,
        expected_gpu_uuid=args.expected_gpu_uuid,
    )
    if args.command == "cell":
        run_cell(**values, method=args.method, gamma=args.gamma, device=args.device)
    elif args.command == "aggregate":
        aggregate(**values)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
