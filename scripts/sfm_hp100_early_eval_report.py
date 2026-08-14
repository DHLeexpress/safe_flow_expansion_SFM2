#!/usr/bin/env python3
"""Aggregate four fixed-bank HP100 raw evaluations without re-evaluation.

The input files must be canonical ``sfm_hp100_eval.py`` outputs for rounds
0--3.  This script is intentionally posthoc-only: it validates the common CRN
and scientific source contracts, checks that the stored summaries agree with
the episode rows, and then renders the requested tables and four-metric plot.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STATUS = "SFM_HP100_RAW_EVAL_COMPLETE"
ROUNDS = (0, 1, 2, 3)
METRICS = (
    "SR",
    "CR",
    "Validity",
    "successful_clearance",
    "successful_time_to_goal",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read evaluation JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"evaluation root must be an object: {path}")
    return value


def _source_contract(payload: dict[str, Any]) -> dict[str, Any]:
    verifier = payload.get("verifier", {})
    return {
        "status": payload.get("status"),
        "version": payload.get("version"),
        "evaluator_sha256": payload.get("evaluator_sha256"),
        "architecture": payload.get("architecture"),
        "dynamics": payload.get("dynamics"),
        "verifier": {
            "contract": verifier.get("contract"),
            "evaluator_sha256": verifier.get("evaluator_sha256"),
            "polytope_sha256": verifier.get("polytope_sha256"),
        },
        "observation": payload.get("observation"),
        "NFE": payload.get("NFE"),
        "temperature": payload.get("temperature"),
        "semantics": payload.get("semantics"),
    }


def _crn_contract(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene": payload.get("scene"),
        "ep0": payload.get("ep0"),
        "M_per_gamma": payload.get("M_per_gamma"),
        "noise_seed": payload.get("noise_seed"),
        "noise_bank": payload.get("noise_bank"),
    }


def _require_equal(name: str, actual: Any, expected: Any, *, round_i: int) -> None:
    if actual != expected:
        raise ValueError(
            f"round {round_i} {name} does not match round 0: "
            f"{actual!r} != {expected!r}"
        )


def _mean(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return None if not finite else float(np.mean(finite))


def _summary_cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        raise ValueError("evaluation cell is empty")
    success = sum(bool(row.get("success")) for row in rows)
    collision = sum(bool(row.get("collision")) for row in rows)
    timeout = sum(bool(row.get("timeout")) for row in rows)
    if success + collision + timeout != n:
        raise ValueError("success/collision/timeout do not partition an evaluation cell")
    if any(row.get("validity") is None for row in rows):
        raise ValueError("canonical evaluation row is missing window Validity")
    return {
        "n": n,
        "SR": success / n,
        "CR": collision / n,
        "Validity": _mean([row["validity"] for row in rows]),
        "successful_clearance": _mean(
            [row.get("successful_clearance") for row in rows]
        ),
        "successful_time_to_goal": _mean(
            [row.get("time_to_goal") for row in rows]
        ),
    }


def _same_number(actual: Any, expected: Any) -> bool:
    if actual is None or expected is None:
        return actual is expected
    return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)


def _validate_cell(
    stored: dict[str, Any], recomputed: dict[str, Any], *, label: str, round_i: int
) -> None:
    if int(stored.get("n", -1)) != recomputed["n"]:
        raise ValueError(f"round {round_i} {label} stored n is inconsistent with rows")
    for metric in METRICS:
        if not _same_number(stored.get(metric), recomputed[metric]):
            raise ValueError(
                f"round {round_i} {label} stored {metric} is inconsistent with rows"
            )


def _validate_payload(
    payload: dict[str, Any], *, round_i: int, gammas: list[str]
) -> None:
    if payload.get("status") != STATUS:
        raise ValueError(f"round {round_i} is not a completed HP100 raw evaluation")
    if float(payload.get("temperature", float("nan"))) != 1.0:
        raise ValueError(f"round {round_i} is not raw temperature=1 evaluation")
    m = int(payload.get("M_per_gamma", 0))
    ep0 = int(payload.get("ep0", -1))
    if m <= 0 or ep0 < 0:
        raise ValueError(f"round {round_i} has an invalid evaluation bank")
    noise = payload.get("noise_bank")
    if not isinstance(noise, dict):
        raise ValueError(f"round {round_i} is missing the CRN noise-bank contract")
    if noise.get("shape") != [len(gammas), m, 180, 20]:
        raise ValueError(f"round {round_i} CRN noise-bank shape is inconsistent")
    if noise.get("dtype") != "float32" or not noise.get("sha256"):
        raise ValueError(f"round {round_i} CRN noise-bank identity is incomplete")

    rows = payload.get("rows")
    summary = payload.get("summary")
    if not isinstance(rows, list) or not isinstance(summary, dict):
        raise ValueError(f"round {round_i} lacks canonical rows or summary")
    per_gamma = summary.get("per_gamma")
    if not isinstance(per_gamma, dict) or list(per_gamma) != gammas:
        raise ValueError(f"round {round_i} per-gamma summary keys/order changed")
    if len(rows) != len(gammas) * m:
        raise ValueError(f"round {round_i} row count is not gamma_count * M")

    observed: set[tuple[float, int]] = set()
    for row in rows:
        pair = (float(row.get("gamma")), int(row.get("episode", -1)))
        if pair in observed:
            raise ValueError(f"round {round_i} repeats evaluation cell {pair}")
        observed.add(pair)
    expected = {
        (float(gamma), episode)
        for gamma in gammas
        for episode in range(ep0, ep0 + m)
    }
    if observed != expected:
        raise ValueError(f"round {round_i} episode/gamma bank does not match ep0 and M")

    _validate_cell(
        summary.get("pooled", {}), _summary_cell(rows),
        label="pooled", round_i=round_i,
    )
    for gamma in gammas:
        cell_rows = [row for row in rows if float(row["gamma"]) == float(gamma)]
        _validate_cell(
            per_gamma[gamma], _summary_cell(cell_rows),
            label=f"gamma={gamma}", round_i=round_i,
        )


def validate_evaluations(
    inputs: dict[int, Path],
) -> tuple[dict[int, dict[str, Any]], list[str], dict[str, Any], dict[str, Any]]:
    if tuple(sorted(inputs)) != ROUNDS:
        raise ValueError("exactly rounds 0, 1, 2, and 3 are required")
    payloads = {round_i: _load(inputs[round_i]) for round_i in ROUNDS}
    first_per_gamma = payloads[0].get("summary", {}).get("per_gamma", {})
    if not isinstance(first_per_gamma, dict) or not first_per_gamma:
        raise ValueError("round 0 has no per-gamma summary")
    gammas = sorted(first_per_gamma, key=float)
    if list(first_per_gamma) != gammas:
        raise ValueError("round 0 per-gamma summary is not numerically ordered")

    source = _source_contract(payloads[0])
    crn = _crn_contract(payloads[0])
    for round_i in ROUNDS:
        payload = payloads[round_i]
        _validate_payload(payload, round_i=round_i, gammas=gammas)
        _require_equal(
            "source contract", _source_contract(payload), source, round_i=round_i
        )
        _require_equal("CRN contract", _crn_contract(payload), crn, round_i=round_i)
    return payloads, gammas, source, crn


def _table_rows(
    payloads: dict[int, dict[str, Any]], gammas: list[str], inputs: dict[int, Path]
) -> list[dict[str, Any]]:
    rows = []
    for round_i in ROUNDS:
        payload = payloads[round_i]
        cells = [("pooled", None, payload["summary"]["pooled"])]
        cells.extend(
            ("gamma", gamma, payload["summary"]["per_gamma"][gamma])
            for gamma in gammas
        )
        for scope, gamma, cell in cells:
            rows.append({
                "round": round_i,
                "scope": scope,
                "gamma": gamma,
                "n": int(cell["n"]),
                **{metric: cell.get(metric) for metric in METRICS},
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "input_json": str(inputs[round_i].resolve()),
                "input_sha256": _sha256(inputs[round_i]),
            })
    return rows


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _metric_value(cell: dict[str, Any], metric: str) -> float:
    value = cell.get(metric)
    return np.nan if value is None else float(value)


def _render(
    payloads: dict[int, dict[str, Any]], gammas: list[str], output: Path
) -> None:
    rounds = np.asarray(ROUNDS)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(gammas)))
    panels = (
        ("CR", "Collision rate"),
        ("Validity", "Window Validity"),
        ("successful_clearance", "Successful min. clearance [m]"),
        ("successful_time_to_goal", "Successful time to goal [s]"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 7.8))
    for axis, (metric, ylabel) in zip(axes.flat, panels):
        for gamma, color in zip(gammas, colors):
            values = [
                _metric_value(payloads[r]["summary"]["per_gamma"][gamma], metric)
                for r in ROUNDS
            ]
            axis.plot(
                rounds, values, color=color, linewidth=1.0, marker="o",
                markersize=3.0, alpha=0.72, label=rf"$\gamma={gamma}$",
            )
        pooled = [
            _metric_value(payloads[r]["summary"]["pooled"], metric)
            for r in ROUNDS
        ]
        axis.plot(
            rounds, pooled, color="black", linewidth=2.8, marker="o",
            markersize=5.0, label="pooled", zorder=20,
        )
        axis.set_xlabel("Expansion round")
        axis.set_ylabel(ylabel)
        axis.set_xticks(rounds)
        axis.grid(alpha=0.22, linewidth=0.7)
        if metric in {"CR", "Validity"}:
            axis.set_ylim(0.0, 1.0)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925),
        ncol=min(8, len(labels)), frameon=False, fontsize=8,
    )
    first = payloads[0]
    scene = first["scene"].get("scene_profile", "unknown")
    figure.suptitle(
        f"HP100 early expansion: fixed raw temperature=1 CRN evaluation\n"
        f"{scene}, M={first['M_per_gamma']}/gamma",
        y=0.995, fontsize=12,
    )
    figure.subplots_adjust(
        left=0.08, right=0.985, bottom=0.08, top=0.84,
        wspace=0.24, hspace=0.30,
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    figure.savefig(temporary, format=output.suffix.lstrip("."), dpi=200)
    plt.close(figure)
    temporary.replace(output)


def build_report(inputs: dict[int, Path], outdir: Path) -> dict[str, Path]:
    payloads, gammas, source, crn = validate_evaluations(inputs)
    rows = _table_rows(payloads, gammas, inputs)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "hp100_early_raw_r0_r3_metrics.csv"
    json_path = outdir / "hp100_early_raw_r0_r3_metrics.json"
    png_path = outdir / "hp100_early_raw_r0_r3_four_metrics.png"
    pdf_path = outdir / "hp100_early_raw_r0_r3_four_metrics.pdf"
    _write_csv(csv_path, rows)
    _write_json(json_path, {
        "status": "SFM_HP100_EARLY_RAW_REPORT_COMPLETE",
        "rounds": list(ROUNDS),
        "gammas": [float(gamma) for gamma in gammas],
        "source_contract": source,
        "crn_contract": crn,
        "metrics": list(METRICS),
        "rows": rows,
        "plot_semantics": (
            "stored fixed-bank raw temperature=1 evaluations only; per-gamma "
            "thin lines and pooled thick line; no smoothing or re-evaluation"
        ),
    })
    _render(payloads, gammas, png_path)
    _render(payloads, gammas, pdf_path)
    return {"csv": csv_path, "json": json_path, "png": png_path, "pdf": pdf_path}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    for round_i in ROUNDS:
        parser.add_argument(f"--r{round_i}", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args(argv)
    inputs = {round_i: getattr(args, f"r{round_i}") for round_i in ROUNDS}
    outputs = build_report(inputs, args.outdir)
    print(json.dumps({key: str(path.resolve()) for key, path in outputs.items()}))


if __name__ == "__main__":
    main()
