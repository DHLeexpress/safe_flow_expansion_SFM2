#!/usr/bin/env python
"""Soft-cost-weighted multi-candidate distillation archive from BoN-16 sidecars.

ADDITIVE module.  Nothing in the frozen pipeline is edited or monkeypatched;
the exact GREEN verifier (``sfm_metrics2.verify_query``) is imported read-only.

Motivation
----------
``sfm_hp100_bon_distill_collect`` executes the argmin-MPC-cost proposal of a
best-of-16 draw and archives ONLY that one window per step (and only when it
passes the exact full-H10 verifier).  The other fifteen proposals are stored
in the ``bon_pairs.pt`` sidecar and thrown away at training time.  Many of
them are also verifier-valid and only marginally worse under the MPC cost:
that is a whole distribution of safe behavior the r1 update never sees.

This module rebuilds the archive as a SOFT, multi-candidate target.  For every
captured step it certifies all sixteen windows with the exact verifier, keeps
the valid ones, and emits one D+ row per valid candidate carrying a
Boltzmann weight over the MPC cost

    w_j  proportional to  exp(-min(c_j - c_min, clip) / tau)

normalized to sum to 1 over that step's valid candidates.  ``tau`` is not a
hand-tuned knob: it is the global median over steps of the valid-cost spread
``q75(c) - min(c)``, i.e. the natural scale at which candidates of one draw
disagree.  The weight rides in ``prediction_audit["mpc_soft_weight"]`` and is
consumed by the ``mpc_soft`` D+ mass mode in ``sfm_hp100_expansion_update``.

Declared scope and gaps
-----------------------
* Steps whose chosen window failed verification have NO ``bon_archive`` row.
  They are SKIPPED (never reconstructed from raw observations), so every
  emitted row inherits a real, verified-step template row; the skipped count
  is reported.
* Sibling candidates (``attempt=j != choice``) reuse the template's
  ``context`` tensor (identical by construction: same step, same observation),
  its ``mode_tags``, ``bon_controller`` and the non-cost fields of
  ``prediction_audit``.  Those audit fields therefore describe the CHOSEN
  candidate, not the sibling; only ``mpc_cost`` and ``mpc_soft_weight`` are
  per-candidate.  ``verification`` is fully per-candidate (fresh verifier).
* Sibling ``flow_base`` is ``zeros(10,2)``: the sidecar does not store the
  per-proposal latent.  The expansion loss never reads ``flow_base``.
* Sibling ``verification.step_margin`` is NaN (the adapter's one-step nominal
  Hp margin is defined for the EXECUTED action only) and ``hp_eligible`` is
  therefore declared True rather than derived; the chosen candidate keeps the
  template's real ``step_margin``/``hp_eligible``.
* All candidates -- the chosen one included -- are read from the fp16 sidecar
  and widened to fp32, so the chosen row's window can differ from the
  template's fp32 ``candidate`` in the last fp16 bits.  The chosen window is
  re-certified and cross-checked against the template label; the agreement
  rate and the max margin deviation are recorded in the header.

Dedup contract: the trainer's key is
``(lineage, scenario_id, step, attempt, round, block)``.  Siblings differ in
``attempt`` (= the candidate index j), so a step contributes distinct keys.
The raw-obs join key is ``(gamma, scenario_id, step)``, which every sibling
shares with its template -- one shard record serves the whole step.

Usage::

    python sfm_hp100_bon_soft_archive.py \
        --source gpu0=<.../night4/bon_collect_gpu0> \
        --source gpu3=<.../night4/bon_collect_gpu3> \
        --output-dir <.../night6> --workers 48

writes ``<output-dir>/soft_r1_gpu0.pt``, ``soft_r1_gpu3.pt`` and
``SOFT_ARCHIVE_COMPLETE.json``.  ``--block-dir LABEL=DIR`` targets individual
block directories and ``--limit-steps N`` caps the work for smoke tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

_SNAPSHOT = os.path.dirname(os.path.abspath(__file__))
if _SNAPSHOT not in sys.path:
    sys.path.insert(0, _SNAPSHOT)

import _paths  # noqa: F401,E402
import sfm_metrics2 as VERIFY  # noqa: E402  (read-only exact verifier)
import sfm_scene as SS  # noqa: E402

VERSION = "sfm_hp100_bon_soft_archive_v1"
STATUS_ARCHIVE = "SFM2_BON_SOFT_ARCHIVE"
STATUS_RUN = "SFM2_BON_SOFT_ARCHIVE_COMPLETE"

# Packed archive context layout (see sfm_hp100_bon_distill_collect.run_block):
# [0:TOKEN_DIM] policy token, [TOKEN_DIM:TOKEN_DIM+4] robot state,
# then n_ped*2 pedestrian xy followed by n_ped*2 pedestrian velocity.
TOKEN_DIM = 176
STATE_DIM = 4
STATE_OFFSET = TOKEN_DIM
PED_OFFSET = TOKEN_DIM + STATE_DIM

DEFAULT_CLIP = 50.0
DEFAULT_MAX_ROWS = 2_500_000
DEFAULT_TARGET_ROWS = 2_200_000
SUBSAMPLE_SEED = 7
# Chosen-window re-certification against the template label must agree at
# least this often; below it the fp16 sidecar is not a faithful stand-in.
CHOSEN_AGREEMENT_FLOOR = 0.99


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


SOURCE_SHA = sha256_file(os.path.abspath(__file__))


def unpack_context(context) -> tuple:
    """Split a packed archive context into (state4, ped_xy, ped_vel).

    The pedestrian count is derived from the packed width rather than
    assumed, so the same code serves any scene profile.
    """
    flat = np.asarray(context, dtype=np.float32).reshape(-1)
    if flat.size < PED_OFFSET or (flat.size - PED_OFFSET) % 4:
        raise ValueError(f"unexpected packed context width {flat.size}")
    n_ped = (flat.size - PED_OFFSET) // 4
    state = flat[STATE_OFFSET:STATE_OFFSET + STATE_DIM].copy()
    ped_xy = flat[PED_OFFSET:PED_OFFSET + 2 * n_ped].reshape(n_ped, 2).copy()
    ped_vel = flat[PED_OFFSET + 2 * n_ped:].reshape(n_ped, 2).copy()
    return state, ped_xy, ped_vel


def _verify_step_worker(payload):
    """Certify every candidate window of ONE step with the exact verifier.

    Identical semantics to ``sfm_metrics2.verify_in_worker`` (it calls the
    same ``verify_query``); the per-step batching and the compact return
    exist only to keep the pool's IPC from dominating: the full result dict
    carries the face objects and the pedestrian prediction tensor, which are
    megabytes per step and are never used here.

    Returns ``(step_id, [(j, valid, slack, progress, errored), ...])`` where
    ``progress`` is the plan's goal progress read off the verifier's own
    integrated segment (``||p_0 - goal|| - ||p_H - goal||``).
    """
    step_id, state, ped_xy, ped_vel, gamma, windows = payload
    goal = np.asarray(SS.GOAL, dtype=float)
    out = []
    for j in range(len(windows)):
        result = VERIFY.verify_query(
            state, windows[j].astype(np.float32), ped_xy, ped_vel, float(gamma),
        )
        if not result.get("resolved", False):
            out.append((int(j), 0, float("nan"), float("nan"), True))
            continue
        segment = np.asarray(result["segment"], dtype=float)
        progress = float(
            np.linalg.norm(segment[0] - goal) - np.linalg.norm(segment[-1] - goal)
        )
        diagnostics = result.get("diagnostics") or {}
        out.append((
            int(j), int(result["y"]),
            float(diagnostics.get("slack", float("nan"))), progress, False,
        ))
    return int(step_id), out


def _parse_labelled(values, *, glob_blocks: bool):
    """Parse ``LABEL=PATH`` arguments into an ordered [(label, block_dir)]."""
    items = []
    for value in values or ():
        if "=" not in value:
            raise ValueError(f"expected LABEL=PATH, got {value!r}")
        label, path = value.split("=", 1)
        root = Path(path).expanduser().resolve()
        if glob_blocks:
            blocks = sorted(root.glob("block_*"))
            if not blocks:
                raise FileNotFoundError(f"no block_* directories under {root}")
            items.extend((label, block) for block in blocks)
        else:
            items.append((label, root))
    return items


def load_steps(block_items, *, limit_steps: int) -> tuple:
    """Join every block's pairs sidecar with its verified-positive rows.

    A pairs entry whose ``(gamma, scenario_id, step)`` key has no archive row
    (the chosen window failed the verifier) is skipped and counted.
    """
    import torch

    steps, sources = [], []
    skipped = 0
    for label, block_dir in block_items:
        archive_path = block_dir / "bon_archive.pt"
        pairs_path = block_dir / "bon_pairs.pt"
        archive = torch.load(archive_path, map_location="cpu", weights_only=False)
        pairs = torch.load(pairs_path, map_location="cpu", weights_only=False)
        by_key = {}
        for row in archive["rows"]:
            key = (f"{float(row['gamma']):g}", int(row["scenario_id"]),
                   int(row["step"]))
            by_key[key] = row
        used = 0
        for pair in pairs["pairs"]:
            key = (str(pair["key"][0]), int(pair["key"][1]), int(pair["key"][2]))
            template = by_key.get(key)
            if template is None:
                skipped += 1
                continue
            if not used:
                # Fail closed on the packed-context layout before spending
                # hours in the verifier pool: the declared pedestrian count
                # of the block must be the one the packing implies.
                packed = np.asarray(template["context"]).reshape(-1).size
                implied = (packed - PED_OFFSET) // 4
                declared = int(archive["config"]["n_ped"])
                if (packed - PED_OFFSET) % 4 or implied != declared:
                    raise ValueError(
                        f"packed context width {packed} implies {implied} "
                        f"pedestrians but {block_dir} declares {declared}"
                    )
            steps.append(dict(
                label=str(label),
                order=(str(label), str(block_dir), key),
                template=template,
                windows=np.asarray(pair["windows"].numpy(), dtype=np.float16),
                costs=np.asarray(pair["costs"].numpy(), dtype=np.float32),
                choice=int(pair["choice"]),
            ))
            used += 1
            if limit_steps and len(steps) >= limit_steps:
                break
        sources.append(dict(
            label=str(label), block_dir=str(block_dir),
            archive_rows=len(archive["rows"]), pairs=len(pairs["pairs"]),
            steps_kept=used,
        ))
        if limit_steps and len(steps) >= limit_steps:
            break
    # Deterministic global order regardless of filesystem listing order.
    steps.sort(key=lambda item: item["order"])
    return steps, sources, skipped


def certify_steps(steps, *, workers: int, chunksize: int, progress_every: int):
    """Exact-verify all 16 windows of every step through a spawn pool."""
    payloads = []
    for index, step in enumerate(steps):
        state, ped_xy, ped_vel = unpack_context(step["template"]["context"])
        payloads.append((
            index, state, ped_xy, ped_vel,
            float(step["template"]["gamma"]),
            np.asarray(step["windows"], dtype=np.float32),
        ))
    stats = dict(
        verified=0, errors=0, chosen_recheck=0, chosen_agree=0,
        chosen_margin_max_abs_dev=0.0,
    )
    context = mp.get_context("spawn")
    started = time.time()
    with context.Pool(int(workers)) as pool:
        for done, (index, results) in enumerate(pool.imap_unordered(
                _verify_step_worker, payloads, chunksize=int(chunksize)), 1):
            step = steps[index]
            valid = []
            for j, y, slack, progress, errored in results:
                stats["errors"] += int(bool(errored))
                if y:
                    valid.append((int(j), float(slack), float(progress)))
                if j == step["choice"]:
                    stats["chosen_recheck"] += 1
                    stats["chosen_agree"] += int(bool(y))
                    template_margin = float(
                        step["template"]["verification"]["margin"]
                    )
                    if np.isfinite(slack) and np.isfinite(template_margin):
                        stats["chosen_margin_max_abs_dev"] = max(
                            stats["chosen_margin_max_abs_dev"],
                            abs(slack - template_margin),
                        )
            step["valid"] = valid
            stats["verified"] += 1
            if progress_every and done % progress_every == 0:
                rate = done / max(1e-9, time.time() - started)
                print(
                    f"[certify] {done}/{len(payloads)} steps "
                    f"({rate:.1f} steps/s, eta "
                    f"{(len(payloads) - done) / max(1e-9, rate) / 60.0:.1f} min)",
                    flush=True,
                )
    return stats


def compute_tau(steps) -> dict:
    """Global median valid-cost spread ``q75 - min`` over steps.

    The median over ALL steps with at least one valid candidate is the
    declared definition.  Single-valid steps contribute a zero spread, so if
    they dominate the median can collapse; the fallback (median over steps
    with at least two valid candidates, then 1.0) is recorded whenever it is
    taken rather than silently applied.
    """
    spreads_all, spreads_multi = [], []
    for step in steps:
        valid = step.get("valid") or []
        if not valid:
            continue
        costs = np.asarray(
            [float(step["costs"][j]) for j, _, _ in valid], dtype=np.float64,
        )
        spread = float(np.percentile(costs, 75) - costs.min())
        spreads_all.append(spread)
        if costs.size >= 2:
            spreads_multi.append(spread)
    median_all = float(np.median(spreads_all)) if spreads_all else 0.0
    median_multi = float(np.median(spreads_multi)) if spreads_multi else 0.0
    tau, source = median_all, "median_spread_all_valid_steps"
    if not (tau > 0.0):
        tau, source = median_multi, "median_spread_multi_valid_steps"
    if not (tau > 0.0):
        tau, source = 1.0, "unit_fallback"
    return dict(
        tau=float(tau), tau_source=source,
        median_spread_all_valid_steps=median_all,
        median_spread_multi_valid_steps=median_multi,
        steps_with_valid=len(spreads_all),
        steps_with_multi_valid=len(spreads_multi),
    )


def step_weights(step, *, tau: float, clip: float) -> np.ndarray:
    """Boltzmann weights over one step's valid candidates (sum exactly 1)."""
    costs = np.asarray(
        [float(step["costs"][j]) for j, _, _ in step["valid"]], dtype=np.float64,
    )
    delta = np.minimum(costs - costs.min(), float(clip))
    weights = np.exp(-delta / float(tau))
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        # Cannot happen (the c_min entry contributes exactly 1.0) but a
        # silently degenerate weight vector would be worse than a hard stop.
        raise FloatingPointError("degenerate soft weight vector")
    return weights / total


def emit_rows(steps, *, tau: float, clip: float):
    """One D+ row per valid candidate, grouped by source label.

    Tensor storage is banked deliberately: the per-group ``candidate``,
    ``context`` and chosen-``flow_base`` tensors are views into three
    contiguous banks, and every sibling of a step shares the one ``context``
    view.  ``torch.save`` serializes a storage once, so a two-million-row
    archive becomes a handful of tensor records instead of millions of them
    (the naive layout makes both the save and every later load pathological).
    ``torch.stack`` over views is unaffected.
    """
    import torch

    zero_flow = torch.zeros(10, 2, dtype=torch.float32)
    buckets: dict[str, dict] = {}
    for step in steps:
        valid = step.get("valid") or []
        if not valid:
            continue
        weights = step_weights(step, tau=tau, clip=clip)
        template = step["template"]
        template_verification = template["verification"]
        template_audit = template["prediction_audit"]
        choice = int(step["choice"])
        bucket = buckets.setdefault(step["label"], dict(
            rows=[], candidates=[], contexts=[], flows=[],
            context_index=[], flow_index=[],
        ))
        context_slot = len(bucket["contexts"])
        bucket["contexts"].append(
            np.asarray(template["context"], dtype=np.float32).reshape(-1)
        )
        for (j, slack, progress), weight in zip(valid, weights):
            chosen = int(j) == choice
            row = dict(template)
            row["attempt"] = int(j)
            row["K_index"] = int(j)
            row["B_local"] = int(j)
            row["role"] = "positive"
            row["negative_reason"] = None
            row["verification"] = dict(
                valid=True,
                hp_eligible=(
                    bool(template_verification["hp_eligible"]) if chosen else True
                ),
                margin=float(slack),
                native_cost=float("nan"),
                H10_progress=float(progress),
                progress_eligible=True,
                error=False,
                step_margin=(
                    float(template_verification["step_margin"])
                    if chosen else float("nan")
                ),
            )
            row["prediction_audit"] = dict(
                template_audit,
                mpc_cost=float(step["costs"][j]),
                mpc_soft_weight=float(weight),
            )
            row["template_source_sha256"] = template.get("source_sha256")
            row["source_sha256"] = SOURCE_SHA
            bucket["candidates"].append(
                np.asarray(step["windows"][j], dtype=np.float32)
            )
            bucket["context_index"].append(context_slot)
            # flow_base is a declared gap for siblings (the sidecar stores no
            # per-proposal latent); the expansion loss never reads it.
            if chosen:
                bucket["flow_index"].append(len(bucket["flows"]))
                bucket["flows"].append(
                    np.asarray(template["flow_base"], dtype=np.float32)
                )
            else:
                bucket["flow_index"].append(-1)
            bucket["rows"].append(row)

    grouped: dict[str, list] = {}
    for label, bucket in buckets.items():
        candidate_bank = torch.from_numpy(
            np.stack(bucket["candidates"]).astype(np.float32, copy=False)
        )
        context_bank = torch.from_numpy(
            np.stack(bucket["contexts"]).astype(np.float32, copy=False)
        )
        flow_bank = (
            torch.from_numpy(
                np.stack(bucket["flows"]).astype(np.float32, copy=False)
            ) if bucket["flows"] else None
        )
        context_views = [context_bank[index] for index in range(len(bucket["contexts"]))]
        for index, row in enumerate(bucket["rows"]):
            row["candidate"] = candidate_bank[index]
            row["context"] = context_views[bucket["context_index"][index]]
            slot = bucket["flow_index"][index]
            row["flow_base"] = zero_flow if slot < 0 else flow_bank[slot]
        grouped[label] = bucket["rows"]
        bucket["candidates"] = bucket["contexts"] = bucket["flows"] = None
    return grouped


def subsample(steps, *, max_rows: int, target_rows: int) -> dict:
    """Cap the emitted-row budget by dropping whole steps, seed-7 shuffled."""
    total = sum(len(step.get("valid") or []) for step in steps)
    report = dict(
        rows_before=int(total), steps_before=len(steps),
        subsampled=False, step_fraction=1.0, row_fraction=1.0,
        max_rows=int(max_rows), target_rows=int(target_rows),
    )
    if total <= int(max_rows):
        return report
    order = list(range(len(steps)))
    random.Random(SUBSAMPLE_SEED).shuffle(order)
    keep, running = set(), 0
    for index in order:
        count = len(steps[index].get("valid") or [])
        if running + count > int(target_rows):
            continue
        keep.add(index)
        running += count
    for index, step in enumerate(steps):
        if index not in keep:
            step["valid"] = []
    report.update(
        subsampled=True, rows_after=int(running), steps_after=len(keep),
        step_fraction=len(keep) / max(1, len(steps)),
        row_fraction=running / max(1, total),
    )
    return report


def build(args) -> dict:
    import torch

    started = time.time()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    block_items = (
        _parse_labelled(args.source, glob_blocks=True)
        + _parse_labelled(args.block_dir, glob_blocks=False)
    )
    if not block_items:
        raise ValueError("at least one --source or --block-dir is required")

    steps, sources, skipped_steps = load_steps(
        block_items, limit_steps=int(args.limit_steps or 0),
    )
    if not steps:
        raise RuntimeError("no joinable steps found")
    print(
        f"[load] {len(steps)} joined steps from {len(block_items)} blocks; "
        f"{skipped_steps} steps skipped (no verified archive row)", flush=True,
    )

    certify = certify_steps(
        steps, workers=int(args.workers), chunksize=int(args.chunksize),
        progress_every=int(args.progress_every),
    )
    agreement = (
        certify["chosen_agree"] / max(1, certify["chosen_recheck"])
    )
    if agreement < CHOSEN_AGREEMENT_FLOOR:
        raise RuntimeError(
            "fp16 sidecar re-certification disagrees with the archive label "
            f"on the chosen window too often: agreement {agreement:.4f}"
        )

    tau_report = compute_tau(steps)
    tau = float(args.tau) if args.tau else float(tau_report["tau"])
    if args.tau:
        tau_report["tau_override"] = float(args.tau)
    clip = float(args.clip)
    print(f"[tau] {tau_report} -> tau={tau:g} clip={clip:g}", flush=True)

    sample_report = subsample(
        steps, max_rows=int(args.max_rows), target_rows=int(args.target_rows),
    )
    print(f"[subsample] {sample_report}", flush=True)

    grouped = emit_rows(steps, tau=tau, clip=clip)
    candidates_total = 16 * len(steps)
    valid_total = sum(len(rows) for rows in grouped.values())

    template_headers: dict[str, dict] = {}
    for step in steps:
        template_headers.setdefault(step["label"], None)
    for label, block_dir in block_items:
        if template_headers.get(label) is None:
            payload = torch.load(
                block_dir / "bon_archive.pt", map_location="cpu",
                weights_only=False,
            )
            template_headers[label] = {
                key: value for key, value in payload.items() if key != "rows"
            }

    outputs = []
    for label in sorted(grouped):
        rows = grouped[label]
        header = dict(template_headers[label])
        header["status"] = STATUS_ARCHIVE
        header["version"] = VERSION
        header["soft_archive"] = dict(
            tau=float(tau), clip=float(clip),
            steps=int(sum(1 for step in steps
                          if step["label"] == label and step.get("valid"))),
            rows=len(rows),
            skipped_steps=int(skipped_steps),
            source_sha256=SOURCE_SHA,
            template_status=str(template_headers[label].get("status")),
            weight_rule=(
                "w_j proportional to exp(-min(c_j - c_min, clip)/tau) over the "
                "exact-verifier-valid candidates of one step; sums to 1 per "
                "step; carried in prediction_audit['mpc_soft_weight']"
            ),
            tau_rule=tau_report,
            declared_gaps=(
                "siblings share the template's context/mode_tags/"
                "bon_controller and every prediction_audit field except "
                "mpc_cost and mpc_soft_weight; sibling flow_base is zeros "
                "(no per-proposal latent in the sidecar) and sibling "
                "step_margin is NaN with hp_eligible declared True; all "
                "candidate windows come from the fp16 sidecar widened to fp32"
            ),
            chosen_recheck=dict(
                checked=int(certify["chosen_recheck"]),
                agreed=int(certify["chosen_agree"]),
                agreement=float(agreement),
                margin_max_abs_dev=float(certify["chosen_margin_max_abs_dev"]),
            ),
            subsample=sample_report,
            blocks=[item for item in sources if item["label"] == label],
        )
        header["rows"] = rows
        path = output_dir / f"soft_r1_{label}.pt"
        torch.save(header, path)
        outputs.append(dict(
            label=label, path=str(path), rows=len(rows),
            bytes=int(path.stat().st_size), sha256=sha256_file(path),
        ))
        print(f"[write] {path} rows={len(rows)}", flush=True)

    marker = dict(
        status=STATUS_RUN, version=VERSION, source_sha256=SOURCE_SHA,
        blocks=len(block_items), sources=sources,
        joined_steps=len(steps), skipped_steps=int(skipped_steps),
        candidates_verified=int(candidates_total),
        verifier_errors=int(certify["errors"]),
        valid_candidates=int(sum(
            len(step.get("valid") or []) for step in steps
        )),
        valid_fraction=float(valid_total) / max(1, candidates_total),
        emitted_rows=int(valid_total),
        tau=float(tau), clip=float(clip), tau_report=tau_report,
        subsample=sample_report,
        chosen_recheck=dict(
            checked=int(certify["chosen_recheck"]),
            agreed=int(certify["chosen_agree"]),
            agreement=float(agreement),
            margin_max_abs_dev=float(certify["chosen_margin_max_abs_dev"]),
        ),
        outputs=outputs, workers=int(args.workers),
        elapsed_seconds=float(time.time() - started),
        verifier_manifest=VERIFY.verifier_manifest(),
    )
    with open(output_dir / args.marker_name, "w") as stream:
        json.dump(marker, stream, indent=2, sort_keys=True, default=str)
    return marker


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--source", action="append", default=[],
        help="LABEL=DIR whose block_* subdirectories are all used",
    )
    value.add_argument(
        "--block-dir", action="append", default=[],
        help="LABEL=BLOCKDIR for a single block directory",
    )
    value.add_argument("--output-dir", required=True)
    value.add_argument("--marker-name", default="SOFT_ARCHIVE_COMPLETE.json")
    value.add_argument("--workers", type=int, default=8)
    value.add_argument("--chunksize", type=int, default=8)
    value.add_argument("--progress-every", type=int, default=2000)
    value.add_argument("--clip", type=float, default=DEFAULT_CLIP)
    value.add_argument(
        "--tau", type=float, default=0.0,
        help="override the measured global tau (0 = measure it)",
    )
    value.add_argument(
        "--limit-steps", type=int, default=0,
        help="debug: stop after this many joined steps",
    )
    value.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    value.add_argument("--target-rows", type=int, default=DEFAULT_TARGET_ROWS)
    return value


def main(argv=None) -> int:
    marker = build(parser().parse_args(argv))
    print({
        "status": marker["status"],
        "joined_steps": marker["joined_steps"],
        "skipped_steps": marker["skipped_steps"],
        "emitted_rows": marker["emitted_rows"],
        "valid_fraction": marker["valid_fraction"],
        "tau": marker["tau"],
        "outputs": [item["path"] for item in marker["outputs"]],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
