"""One-shot raw-observation training from r0 on extended archives.

Loads merged/tagged extended archive rows plus their raw-observation shards,
joins them fail-closed, and runs a single declared ``expansion_update`` from
the canonical r0 with the raw-forward context provider, so the widened
``trunk_head_and_projection`` scope can flow gradients into
``grid_projection`` (archived tokens alone cannot reach any encoder).  Every
existing declared knob — alpha with literal/hinge negative modes, the D+
mass modes including ``mode_gamma_tree``, exposure passes, clip/drift/finite
guards, exposure and mass audits — runs unchanged through the provider path.

``--subset-fraction`` applies the stratified-nested convention (per-gamma
``random.Random(7)`` shuffle, prefix of each gamma group, per role), so
fraction ``k/10`` reproduces the nested pseudo-round subsets and any two
fractions are strictly nested.

Outputs, under a fresh ``--output`` directory: ``checkpoint_r1.pt`` (strict
raw-evaluable schema, written only when the update is accepted) and
``RAW_TRAIN_COMPLETE.json`` with the full audit, the token re-encode audit,
and the grid_projection drift reported separately from the trunk/head drift.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_ball_launch as BASE
import sfm_hp100_exhaustive_hybrid as HYBRID
import sfm_hp100_expansion_update as UPD
import sfm_hp100_predictive_execution as PRED
import sfm_hp100_raw_obs_dataset as RDS

VERSION = "sfm_hp100_raw_train_v1"
STATUS = "SFM2_RAW_TRAIN_COMPLETE"
SUBSET_SHUFFLE_SEED = 7


def stratified_prefix(rows, fraction: float):
    """Per-gamma seed-7 shuffle + prefix: the stratified-nested convention."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("subset fraction must lie in (0,1]")
    if fraction == 1.0:
        return list(rows)
    by_gamma = defaultdict(list)
    for row in rows:
        by_gamma[float(row["gamma"])].append(row)
    subset = []
    for gamma in sorted(by_gamma):
        group = by_gamma[gamma]
        random.Random(SUBSET_SHUFFLE_SEED).shuffle(group)
        take = max(1, int(len(group) * fraction))
        subset.extend(group[:take])
    return subset


def _load_archives(paths):
    rows, sources = [], []
    for path in paths:
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        sources.append({
            "path": str(path.resolve()),
            "sha256": PRED.sha256_file(path),
            "rows": len(payload["rows"]),
        })
        rows.extend(payload["rows"])
    return rows, sources


def _per_gamma_counts(rows) -> dict:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[f"{float(row['gamma']):g}"] += 1
    return dict(sorted(counts.items(), key=lambda item: float(item[0])))


def run(args) -> dict:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing raw-train output: {output}")
    gpu = BASE._gpu_contract(args.device, int(args.physical_gpu))
    r0_sha = PRED.sha256_file(args.checkpoint)
    if r0_sha != str(args.expected_checkpoint_sha256).lower():
        raise RuntimeError("r0 checkpoint SHA256 mismatch")
    policy, _ = GPS.load_sfm_hp100_policy(args.checkpoint, device=args.device)
    adapter = PORT.HP100ExpansionPolicy(policy).eval()

    config = UPD.UpdateConfig(
        alpha=float(args.alpha),
        negative_mode=str(args.negative_mode),
        negative_margin=float(args.negative_margin),
        learning_rate=float(args.learning_rate),
        batch_size=int(args.batch_size),
        exposure_passes=int(args.exposure_passes),
        grad_clip_norm=float(args.grad_clip_norm),
        max_relative_parameter_drift=float(args.max_relative_parameter_drift),
        train_mode=str(args.train_mode),
        optimizer_scope=str(args.optimizer_scope),
        positive_mass=str(args.positive_mass),
        avoid_mass=float(args.avoid_mass),
        seed=int(args.update_seed),
    )
    config.validate()

    rows, archive_sources = _load_archives(args.archive)
    positives = [row for row in rows if row["role"] == "positive"]
    negatives = [row for row in rows if row["role"] != "positive"]
    positives = stratified_prefix(positives, float(args.subset_fraction))
    negatives = stratified_prefix(negatives, float(args.subset_fraction))

    dataset = None
    provider = None
    token_audit = None
    if args.context_path == "raw":
        if not args.raw_manifest:
            raise ValueError(
                "--context-path raw requires at least one --raw-manifest"
            )
        dataset = RDS.RawObsDataset(
            args.raw_manifest, lru_shards=int(args.lru_shards),
        )
        dataset.assert_joined(positives + negatives)
        audit_rows = (positives + negatives)[::max(1, int(args.raw_audit_every))]
        token_audit = RDS.audit_tokens(
            dataset, adapter, audit_rows, atol=float(args.raw_audit_atol),
        )
        if not token_audit["ok"]:
            raise RuntimeError(
                "raw-obs token audit failed: max deviation "
                f"{token_audit['max_abs_deviation']} > atol {token_audit['atol']}"
            )
        provider = RDS.make_context_provider(dataset, adapter)

    # Per-group drift: the projection surface is audited separately from the
    # trunk/head surface (both restored together on reject).
    UPD.configure_trainable(adapter, config.optimizer_scope)
    groups: dict[str, list[torch.nn.Parameter]] = {
        "trunk_head": [], "grid_projection": [],
    }
    for name, parameter in adapter.named_parameters():
        if not parameter.requires_grad:
            continue
        key = (
            "grid_projection" if name.startswith("policy.grid_projection.")
            else "trunk_head"
        )
        groups[key].append(parameter)
    snapshots = {
        key: HYBRID._parameter_snapshot(parameters)
        for key, parameters in groups.items() if parameters
    }

    metrics = UPD.expansion_update(
        adapter, positives, negatives, config,
        round_index=1, context_provider=provider,
    )
    drift_by_group = {
        key: float(HYBRID._relative_parameter_drift(
            groups[key], snapshots[key],
        ))
        for key in snapshots
    }

    output.mkdir(parents=True)
    checkpoint = None
    if metrics["accepted"]:
        payload = UPD.checkpoint_payload(
            adapter,
            parent_checkpoint_sha256=r0_sha,
            pretrained_checkpoint_sha256=r0_sha,
            round_index=1,
            alpha=config.alpha,
            exposure_passes=config.exposure_passes,
            optimizer_scope=config.optimizer_scope,
        )
        checkpoint_path = output / "checkpoint_r1.pt"
        torch.save(payload, checkpoint_path)
        checkpoint = {
            "path": str(checkpoint_path),
            "sha256": PRED.sha256_file(checkpoint_path),
        }

    marker = {
        "status": STATUS,
        "version": VERSION,
        "gpu": gpu,
        "r0": {"path": str(Path(args.checkpoint).resolve()), "sha256": r0_sha},
        "archives": archive_sources,
        "raw_manifests": (
            None if dataset is None else {
                "directories": [str(d) for d in dataset.directories],
                "rows": len(dataset),
                "hp_roundtrip_max_abs": dataset.hp_roundtrip_max_abs,
            }
        ),
        "context_path": args.context_path,
        "token_audit": token_audit,
        "subset_fraction": float(args.subset_fraction),
        "subset_convention": (
            f"stratified-nested: per-gamma Random({SUBSET_SHUFFLE_SEED}) "
            "shuffle, per-gamma prefix, per role"
        ),
        "positive_rows": len(positives),
        "negative_rows": len(negatives),
        "positive_rows_per_gamma": _per_gamma_counts(positives),
        "update": metrics,
        "relative_drift_by_group": drift_by_group,
        "checkpoint": checkpoint,
    }
    PRED._write_json(output / "RAW_TRAIN_COMPLETE.json", marker)
    return marker


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--expected-checkpoint-sha256", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--archive", action="append", required=True)
    value.add_argument("--raw-manifest", action="append", default=[])
    value.add_argument(
        "--context-path", choices=("raw", "stored"), default="raw",
    )
    value.add_argument("--subset-fraction", type=float, default=1.0)
    value.add_argument("--lru-shards", type=int, default=RDS.DEFAULT_LRU_SHARDS)
    value.add_argument("--raw-audit-atol", type=float, default=1.0e-3)
    value.add_argument("--raw-audit-every", type=int, default=500)
    value.add_argument("--alpha", type=float, default=0.0)
    value.add_argument(
        "--negative-mode", choices=("literal", "hinge"), default="literal",
    )
    value.add_argument("--negative-margin", type=float, default=2.0)
    value.add_argument("--learning-rate", type=float, default=1.0e-5)
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--exposure-passes", type=int, default=1)
    value.add_argument(
        "--optimizer-scope",
        choices=sorted(UPD.DECLARED_TRAINABLE_SURFACES),
        default=UPD.PROJECTION_OPTIMIZER_SCOPE,
    )
    value.add_argument(
        "--positive-mass", choices=UPD.POSITIVE_MASS_MODES,
        default="per_gamma_balanced",
    )
    value.add_argument("--avoid-mass", type=float, default=0.65)
    value.add_argument("--grad-clip-norm", type=float, default=1.0)
    value.add_argument(
        "--max-relative-parameter-drift", type=float, default=0.25,
    )
    value.add_argument("--train-mode", choices=("eval", "train"), default="eval")
    value.add_argument("--update-seed", type=int, default=2)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--physical-gpu", type=int, required=True)
    return value


def main(argv=None) -> int:
    marker = run(parser().parse_args(argv))
    print({
        "status": marker["status"],
        "accepted": marker["update"]["accepted"],
        "positive_rows": marker["positive_rows"],
        "drift_by_group": marker["relative_drift_by_group"],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
