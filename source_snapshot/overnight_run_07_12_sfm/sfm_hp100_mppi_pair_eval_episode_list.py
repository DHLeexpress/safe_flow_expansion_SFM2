#!/usr/bin/env python
"""Explicit-episode-list driver for the locked MPPI-DCBF expert (ADDITIVE).

``sfm_hp100_mppi_pair_eval.py`` is the frozen expert-v2 average-clearance
evaluator; its CLI can only sweep a CONTIGUOUS bank (``--ep0``/``--M``).  The
v2 "true random scenario" wave needs an arbitrary, per-gamma, disjoint list of
scenario ids instead.  This module does NOT modify or monkeypatch that
evaluator: it imports it and reuses, verbatim,

  * ``rollout_controller_episode``  -- the closed-loop rollout and the
    terminal-inclusive ``episode_average_clearance`` accounting,
  * ``make_planner``               -- the locked ``CappedSafeMPPIAdapter``,
  * ``sfm_hp100_eval.attach_validity`` -- the frozen exact GREEN verifier,
  * ``summarize_selected``         -- the cell summary,
  * ``build_payload``              -- the full provenance payload (evaluator
    path/SHA, git state, scene, controller config, verifier manifest, metric
    semantics),

and only replaces the episode iterator.  Rows are therefore bit-identical in
construction to a stock v2 shard; only WHICH scenes are visited differs.

The emitted payload keeps every stock ``sfm_hp100_mppi_pair_eval_v2`` block and
adds, additively:

  status                = SFM_HP100_MPPI_PAIR_EVAL_EPISODE_LIST_COMPLETE
                          (deliberately distinct: this is not a contiguous-bank
                          shard and must not validate as one)
  base_status/version   = the stock values it derives from
  wrapper_source/_sha256, wave, episodes, episode_list_json/_sha256
  bank.episode_source   = "declared_episode_list"
  cluster_summary       = the learned-cell ``summary`` block
  rows[i].J             = None          (no best-of-J selection here)
  rows[i].episode_min_clearance = rows[i].min_clearance   (name mapping to the
                          learned-cell schema; identical quantity)
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import types

DEFAULT_SNAPSHOT = os.path.dirname(os.path.abspath(__file__))


def _bootstrap(snapshot: str):
    snapshot = os.path.abspath(snapshot)
    if snapshot not in sys.path:
        sys.path.insert(0, snapshot)
    import _paths  # noqa: F401


STATUS = "SFM_HP100_MPPI_PAIR_EVAL_EPISODE_LIST_COMPLETE"
WRAPPER_VERSION = "sfm_hp100_mppi_pair_eval_episode_list_v1"


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gamma_tag(gamma: float) -> str:
    text = f"{float(gamma):.4f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return text.replace(".", "p")


def load_episode_list(path, gamma):
    with open(path) as stream:
        raw = json.load(stream)
    numeric = {float(key): value for key, value in raw.items()}
    matches = [k for k in numeric if abs(k - float(gamma)) <= 1e-12]
    if len(matches) != 1:
        raise ValueError(f"episode list has no unique entry for gamma={gamma:g}")
    ids = [int(v) for v in numeric[matches[0]]]
    if not ids or len(set(ids)) != len(ids) or any(v < 0 for v in ids):
        raise ValueError(f"invalid episode list for gamma={gamma:g}")
    return ids


def _mean(values):
    finite = [float(value) for value in values if value is not None]
    return None if not finite else float(sum(finite) / len(finite))


def cluster_summary(rows) -> dict:
    """Exactly the learned-cell ``summary`` block of sfm_hp100_cluster_deploy."""
    n = len(rows)
    successes = sum(bool(row["success"]) for row in rows)
    collisions = sum(bool(row["collision"]) for row in rows)
    timeouts = sum(bool(row["timeout"]) for row in rows)
    if successes + collisions + timeouts != n:
        raise RuntimeError("outcomes do not partition the cell")
    successful = [row for row in rows if row["success"]]
    return dict(
        attempts=int(n), successes=int(successes), collisions=int(collisions),
        timeouts=int(timeouts),
        SR=(successes / n if n else None),
        CR=(collisions / n if n else None),
        timeout_rate=(timeouts / n if n else None),
        all_attempts=dict(
            mean_average_clearance=_mean(
                [row["episode_average_clearance"] for row in rows]),
            mean_min_clearance=_mean(
                [row["episode_min_clearance"] for row in rows]),
        ),
        success_conditioned=dict(
            n=len(successful),
            mean_time_to_goal=_mean([row["time_to_goal"] for row in successful]),
            mean_average_clearance=_mean(
                [row["episode_average_clearance"] for row in successful]),
            mean_min_clearance=_mean(
                [row["episode_min_clearance"] for row in successful]),
        ),
    )


def evaluate_episode_list(PAIR, RAW, *, method, scene_profile, episodes, gamma,
                          device, validity_executor=None):
    """The mppi_dcbf branch of ``PAIR.evaluate`` over an explicit id list."""
    started = time.perf_counter()
    raw_rows = []
    for episode in episodes:
        row = PAIR.rollout_controller_episode(
            PAIR.make_planner(method), method=method,
            scene_profile=scene_profile, episode=int(episode),
            gamma=float(gamma), device=device,
        )
        raw_rows.append(row)
        print(json.dumps(dict(
            event="controller_episode_complete", method=method,
            scene_profile=scene_profile, gamma=float(gamma),
            completed=len(raw_rows), total=len(episodes),
            episode=int(episode), outcome=row["status"], steps=row["steps"],
        )), flush=True)
    controller_seconds = time.perf_counter() - started
    print(json.dumps(dict(
        event="validity_start", method=method, scene_profile=scene_profile,
        metric_rows=len(raw_rows),
    )), flush=True)
    validity_started = time.perf_counter()
    compact = RAW.attach_validity(raw_rows, executor=validity_executor)
    validity_seconds = time.perf_counter() - validity_started
    return dict(
        rows=compact,
        summary=PAIR.summarize_selected(compact, (float(gamma),)),
        controller_reference=None, gammas_evaluated=[float(gamma)],
        controller_gammas_evaluated=[float(gamma)],
        unique_controller_rollouts=len(raw_rows), metric_rows=len(compact),
        verifier_gamma_expansion=False,
        timing_seconds=dict(controller=controller_seconds,
                            validity=validity_seconds,
                            total=time.perf_counter() - started),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--method", default="mppi_dcbf", choices=("mppi_dcbf",))
    parser.add_argument("--scene-profile", default="double_density_velocity_ood")
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--episode-list-json", required=True)
    parser.add_argument("--series", default="dcbf")
    parser.add_argument("--wave", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verifier-workers", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    _bootstrap(args.snapshot)
    import sfm_hp100_eval as RAW
    import sfm_hp100_mppi_pair_eval as PAIR

    if int(args.verifier_workers) <= 0:
        parser.error("--verifier-workers must be positive")
    target = Path(os.path.abspath(args.out))
    temporary = Path(str(target) + ".tmp")
    for candidate in (target, temporary):
        if os.path.lexists(candidate):
            raise FileExistsError(f"refusing to overwrite existing output: {candidate}")

    # The frozen evaluator's own gamma gate, unchanged.
    canonical = PAIR._canonical_gamma(float(args.gamma), include_ood_audit=True)
    PAIR._validate_request(args.method, args.scene_profile, 1, canonical)

    episodes = load_episode_list(args.episode_list_json, canonical)
    episode_list_sha = sha256_file(args.episode_list_json)
    wrapper_source = os.path.abspath(__file__)
    wrapper_sha = sha256_file(wrapper_source)
    print(json.dumps({
        "event": "episode_list_loaded", "path": os.path.abspath(args.episode_list_json),
        "sha256": episode_list_sha, "M": len(episodes), "gamma": canonical,
        "evaluator": os.path.abspath(PAIR.__file__),
        "evaluator_sha256": sha256_file(PAIR.__file__),
    }), flush=True)

    initial_sources = PAIR._source_hashes()
    kwargs = dict(method=args.method, scene_profile=args.scene_profile,
                  episodes=episodes, gamma=canonical, device=args.device)
    if int(args.verifier_workers) == 1:
        result = evaluate_episode_list(PAIR, RAW, **kwargs)
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=int(args.verifier_workers),
                                 mp_context=context) as executor:
            result = evaluate_episode_list(PAIR, RAW, validity_executor=executor,
                                           **kwargs)
    if PAIR._source_hashes() != initial_sources:
        raise RuntimeError("expert-pair evaluator sources changed during evaluation")

    # Reuse the frozen payload builder verbatim, then overwrite exactly the
    # blocks whose contiguous-bank semantics no longer hold.
    shim = types.SimpleNamespace(
        method=args.method, scene_profile=args.scene_profile,
        ep0=min(episodes), M=len(episodes), gamma=canonical,
        include_ood_audit_gamma=False, verifier_workers=int(args.verifier_workers),
    )
    payload = PAIR.build_payload(shim, result, initial_sources)

    for row in payload["rows"]:
        row["J"] = None
        row["episode_min_clearance"] = row["min_clearance"]

    payload.update(
        status=STATUS,
        base_status=PAIR.STATUS,
        wrapper_version=WRAPPER_VERSION,
        wrapper_source=wrapper_source,
        wrapper_sha256=wrapper_sha,
        wave=(None if args.wave is None else str(args.wave)),
        series=str(args.series),
        J=None,
        gamma=canonical,
        gamma_tag=gamma_tag(canonical),
        ep0=None,
        M=len(episodes),
        M_per_gamma=len(episodes),
        episodes=list(episodes),
        episode_source="declared_episode_list",
        episode_list_json=os.path.abspath(args.episode_list_json),
        episode_list_sha256=episode_list_sha,
        cluster_summary=cluster_summary(payload["rows"]),
        cluster_cell_schema_mapping={
            "episode": "identical",
            "gamma": "identical",
            "status": "identical lowercase success/collision/timeout",
            "success": "identical",
            "steps": "identical",
            "time_to_goal": "identical (steps * dt on success, null otherwise)",
            "episode_average_clearance": (
                "identical terminal-inclusive path average; the learned runner "
                "reproduces this same expert-v2 accounting"),
            "episode_min_clearance": "added alias of the evaluator's min_clearance",
            "J": "added as null; MPPI-DCBF performs no best-of-J selection",
            "episode_average_clearance_executed_steps": (
                "NOT emitted by the expert evaluator; absent here"),
        },
    )
    payload["bank"] = dict(
        episode_source="declared_episode_list",
        episode_list_json=os.path.abspath(args.episode_list_json),
        episode_list_sha256=episode_list_sha,
        episodes=list(episodes),
        M=len(episodes),
        contiguous_bank=False,
        same_scenario_ids_for_every_gamma=False,
        pedestrian_seeding="SS.make_humans(episode, seed=0, profile)",
        controller_step_seed="episode * 200 + step",
        controller_seed_pairing_across_gamma=False,
        paired_controller_seed_semantics=(
            "each gamma visits its own disjoint scenario set, so no seed "
            "pairing across gamma exists in this wave"),
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    with open(temporary, "x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    os.replace(temporary, target)
    print(json.dumps({
        "status": payload["status"], "method": payload["method"],
        "gamma": canonical, "out": str(target),
        "metric_rows": payload["metric_rows"],
        "cluster_summary": payload["cluster_summary"],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
