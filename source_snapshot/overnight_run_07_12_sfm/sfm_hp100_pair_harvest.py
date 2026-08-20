"""Harvest same-context (safe, unsafe) action pairs from acquisition traces.

The diagnosed blind spot is same-context tail mass: the policy collides at
contexts where a verified-safe demonstration exists. The direct training
signal for that failure is a *pair* from the same replan: the executed exact
positive versus an exact negative from the same B=32 block. The traces store
per-attempt ``candidate_ids``, ``verification``, ``prediction_audits`` and
the rolled ``B_segments`` positions, but not the raw H10 *action* windows of
the non-selected candidates. Actions are recovered through the deterministic
sampling seam: per-context flow noise depends only on the recorded sampling
seed (``HYBRID._sampling_seed(block_seed, "predictive_always_on", key, step,
microcycle=0, attempt)``) and the retry ``base_std`` — never on batch
composition — so replaying ``ACQ._sample_blocks`` on the stored context
reproduces all K=64 plans up to GPU kernel-order noise. Every replayed pair
is validated against the stored ``B_segments`` positions fail-closed.

Negative choice per executed attempt: among exact-negative locals, the one
the old progress-argmax would have been most tempted by — maximum verifier
H10 progress, ties broken by minimum CV-predicted clearance. Contexts whose
final block had no exact negative yield no pair.

Only ``executed_role == "positive"`` events contribute (realized-failure
executions are excluded from the positive side by construction).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_ball_launch as BASE
import sfm_hp100_early_acquisition as ACQ
import sfm_hp100_exhaustive_hybrid as HYBRID
import sfm_hp100_predictive_execution as PRED

VERSION = "sfm_hp100_pair_harvest_v1"
STATUS = "SFM2_PAIR_ARCHIVE"
SEGMENT_ATOL = 2.0e-3
MAX_MISMATCH_RATE = 0.005


def choose_negative_local(attempt_row: dict) -> int | None:
    """Most tempting exact negative: max progress, then min CV clearance."""
    candidates = []
    for local, verification in enumerate(attempt_row["verification"]):
        if verification.get("error") or verification.get("valid"):
            continue
        audit = attempt_row["prediction_audits"][local]
        candidates.append((
            -float(verification["H10_progress"]),
            float(audit["predicted_min_clearance"]),
            local,
        ))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def executed_pair_specs(events) -> list[dict]:
    """One pair spec per executed-positive event with an exact negative."""
    specs = []
    for event in events:
        if event.get("executed_role") != "positive" or not event["attempts"]:
            continue
        attempt_row = event["attempts"][-1]
        pos_local = attempt_row.get("predictive_local")
        if pos_local is None:
            continue
        neg_local = choose_negative_local(attempt_row)
        if neg_local is None:
            continue
        specs.append({
            "event": event,
            "attempt_row": attempt_row,
            "pos_local": int(pos_local),
            "neg_local": int(neg_local),
        })
    return specs


def _segment_matches(state_before, action, stored_segment, atol) -> bool:
    rolled = PORT.clipped_plan_states(
        np.asarray(state_before, np.float32),
        action.detach().cpu().numpy(),
    )[:, :2].astype(np.float32, copy=False)
    return bool(np.allclose(rolled, stored_segment, atol=atol))


def harvest_trace(
    trace_events,
    adapter: PORT.HP100ExpansionPolicy,
    *,
    block_seed: int,
    source: str,
    device: torch.device,
    replay_chunk: int = 48,
    segment_atol: float = SEGMENT_ATOL,
) -> tuple[list[dict], dict]:
    """Replay and validate pairs for one trace. Fail-closed on mismatch rate."""
    specs = executed_pair_specs(trace_events)
    # Group by (base_std) so each _sample_blocks call is legal, chunked for
    # GPU memory; per-context noise is seed-deterministic so grouping is free.
    by_std: dict[float, list[dict]] = defaultdict(list)
    for spec in specs:
        by_std[float(spec["attempt_row"]["base_std"])].append(spec)
    pairs, mismatches, checked = [], 0, 0
    for base_std, block in sorted(by_std.items()):
        for start in range(0, len(block), replay_chunk):
            chunk = block[start:start + replay_chunk]
            contexts = [
                spec["event"]["context"].to(device) for spec in chunk
            ]
            seeds = [
                HYBRID._sampling_seed(
                    int(block_seed), "predictive_always_on",
                    HYBRID.LineageKey(
                        gamma=float(spec["event"]["gamma"]),
                        replica=int(spec["event"]["replica"]),
                    ),
                    int(spec["event"]["step"]),
                    microcycle=0,
                    attempt=int(spec["attempt_row"]["attempt"]),
                )
                for spec in chunk
            ]
            with torch.inference_mode():
                plans, _, _, _ = ACQ._sample_blocks(
                    adapter, contexts, seeds, K=PRED.K,
                    flow_base_std=float(base_std),
                )
            for spec, plan_block in zip(chunk, plans):
                event = spec["event"]
                attempt_row = spec["attempt_row"]
                ids = attempt_row["candidate_ids"]
                pos_action = plan_block[int(ids[spec["pos_local"]])]
                neg_action = plan_block[int(ids[spec["neg_local"]])]
                checked += 2
                ok = _segment_matches(
                    event["state_before"], pos_action,
                    attempt_row["B_segments"][spec["pos_local"]], segment_atol,
                ) and _segment_matches(
                    event["state_before"], neg_action,
                    attempt_row["B_segments"][spec["neg_local"]], segment_atol,
                )
                if not ok:
                    mismatches += 2
                    continue
                pairs.append({
                    "context": event["context"].detach().cpu().to(torch.float32),
                    "gamma": float(event["gamma"]),
                    "replica": int(event["replica"]),
                    "scenario_id": int(event["scenario_id"]),
                    "step": int(event["step"]),
                    "attempt": int(attempt_row["attempt"]),
                    "pos_action": pos_action.detach().cpu().to(torch.float32).clone(),
                    "neg_action": neg_action.detach().cpu().to(torch.float32).clone(),
                    "pos_local": spec["pos_local"],
                    "neg_local": spec["neg_local"],
                    "pos_verification": dict(
                        attempt_row["verification"][spec["pos_local"]]
                    ),
                    "neg_verification": dict(
                        attempt_row["verification"][spec["neg_local"]]
                    ),
                    "provenance": {"source": source, "block_seed": int(block_seed)},
                })
    stats = {
        "specs": len(specs),
        "pairs": len(pairs),
        "replayed_actions_checked": checked,
        "segment_mismatches": mismatches,
        "mismatch_rate": (mismatches / checked) if checked else 0.0,
    }
    if checked and stats["mismatch_rate"] > MAX_MISMATCH_RATE:
        raise RuntimeError(
            f"pair replay mismatch rate {stats['mismatch_rate']:.4f} exceeds "
            f"{MAX_MISMATCH_RATE} for {source}; refusing unverifiable pairs"
        )
    return pairs, stats


def _collection_block_seed(collection: Path) -> int:
    marker = json.loads((collection / "ARCHIVE_COMPLETE.json").read_text())
    return int(marker["block_summaries"][0]["block_seed"])


def run(args) -> dict:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing pair archive: {output}")
    BASE._gpu_contract(args.device, int(args.physical_gpu))
    checkpoint_sha = PRED.sha256_file(args.checkpoint)
    if checkpoint_sha != str(args.expected_checkpoint_sha256).lower():
        raise RuntimeError("sampling checkpoint SHA256 mismatch")
    policy, _ = GPS.load_sfm_hp100_policy(args.checkpoint, device=args.device)
    adapter = PORT.HP100ExpansionPolicy(policy).eval()
    device = next(adapter.parameters()).device

    all_pairs: list[dict] = []
    per_source = {}
    for collection in map(Path, args.collection):
        block_seed = _collection_block_seed(collection)
        trace = torch.load(
            collection / "trace_block_000.pt",
            map_location="cpu", weights_only=False,
        )
        pairs, stats = harvest_trace(
            trace["events"], adapter,
            block_seed=block_seed, source=str(collection), device=device,
            replay_chunk=int(args.replay_chunk),
            segment_atol=float(args.segment_atol),
        )
        per_source[str(collection)] = stats
        all_pairs.extend(pairs)
        print(f"{collection.name}: {stats}", flush=True)

    if len(all_pairs) < int(args.min_pairs):
        raise RuntimeError(
            f"harvested only {len(all_pairs)} pairs < required {args.min_pairs}"
        )
    per_gamma = defaultdict(int)
    for pair in all_pairs:
        per_gamma[f"{pair['gamma']:g}"] += 1
    payload = {
        "status": STATUS,
        "version": VERSION,
        "sampling_checkpoint": str(args.checkpoint),
        "sampling_checkpoint_sha256": checkpoint_sha,
        "segment_atol": float(args.segment_atol),
        "per_source": per_source,
        "per_gamma": dict(per_gamma),
        "pairs": all_pairs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(json.dumps({"pairs": len(all_pairs), "per_gamma": dict(per_gamma)}))
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--collection", action="append", required=True,
                       help="collection dir with trace_block_000.pt + ARCHIVE_COMPLETE.json")
    value.add_argument("--checkpoint", required=True,
                       help="the SAMPLING policy that gathered the traces (r0)")
    value.add_argument("--expected-checkpoint-sha256", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--min-pairs", type=int, default=10000)
    value.add_argument("--replay-chunk", type=int, default=48)
    value.add_argument("--segment-atol", type=float, default=SEGMENT_ATOL)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--physical-gpu", type=int, required=True)
    return value


def main(argv=None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
