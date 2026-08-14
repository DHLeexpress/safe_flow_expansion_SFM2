#!/usr/bin/env python3
"""Aggregate a variable number of fixed-bank HP100 raw evaluations.

Variable-round SFM2 counterpart of ``sfm_hp100_early_eval_report.py`` (which
is hard-wired to rounds 0-3).  Inputs are canonical ``sfm_hp100_eval.py``
outputs passed as ordered ``--eval label=path`` pairs; the first label is the
baseline reference for the shared source and CRN contracts.  The script is
intentionally posthoc-only: it validates the stored summaries against the
episode rows and renders the four-metric per-gamma tables and plot without any
re-evaluation.
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


def _require_equal(name: str, actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise ValueError(
            f"evaluation {label!r} {name} does not match the baseline: "
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
    stored: dict[str, Any], recomputed: dict[str, Any], *, cell: str, label: str
) -> None:
    if int(stored.get("n", -1)) != recomputed["n"]:
        raise ValueError(f"evaluation {label!r} {cell} stored n is inconsistent with rows")
    for metric in METRICS:
        if not _same_number(stored.get(metric), recomputed[metric]):
            raise ValueError(
                f"evaluation {label!r} {cell} stored {metric} is inconsistent with rows"
            )


def _validate_payload(
    payload: dict[str, Any], *, label: str, gammas: list[str]
) -> None:
    if payload.get("status") != STATUS:
        raise ValueError(f"evaluation {label!r} is not a completed HP100 raw evaluation")
    if float(payload.get("temperature", float("nan"))) != 1.0:
        raise ValueError(f"evaluation {label!r} is not raw temperature=1 evaluation")
    m = int(payload.get("M_per_gamma", 0))
    ep0 = int(payload.get("ep0", -1))
    if m <= 0 or ep0 < 0:
        raise ValueError(f"evaluation {label!r} has an invalid evaluation bank")
    noise = payload.get("noise_bank")
    if not isinstance(noise, dict):
        raise ValueError(f"evaluation {label!r} is missing the CRN noise-bank contract")
    if noise.get("shape") != [len(gammas), m, 180, 20]:
        raise ValueError(f"evaluation {label!r} CRN noise-bank shape is inconsistent")
    if noise.get("dtype") != "float32" or not noise.get("sha256"):
        raise ValueError(f"evaluation {label!r} CRN noise-bank identity is incomplete")

    rows = payload.get("rows")
    summary = payload.get("summary")
    if not isinstance(rows, list) or not isinstance(summary, dict):
        raise ValueError(f"evaluation {label!r} lacks canonical rows or summary")
    per_gamma = summary.get("per_gamma")
    if not isinstance(per_gamma, dict) or list(per_gamma) != gammas:
        raise ValueError(f"evaluation {label!r} per-gamma summary keys/order changed")
    if len(rows) != len(gammas) * m:
        raise ValueError(f"evaluation {label!r} row count is not gamma_count * M")

    observed: set[tuple[float, int]] = set()
    for row in rows:
        pair = (float(row.get("gamma")), int(row.get("episode", -1)))
        if pair in observed:
            raise ValueError(f"evaluation {label!r} repeats evaluation cell {pair}")
        observed.add(pair)
    expected = {
        (float(gamma), episode)
        for gamma in gammas
        for episode in range(ep0, ep0 + m)
    }
    if observed != expected:
        raise ValueError(
            f"evaluation {label!r} episode/gamma bank does not match ep0 and M"
        )

    _validate_cell(
        summary.get("pooled", {}), _summary_cell(rows),
        cell="pooled", label=label,
    )
    for gamma in gammas:
        cell_rows = [row for row in rows if float(row["gamma"]) == float(gamma)]
        _validate_cell(
            per_gamma[gamma], _summary_cell(cell_rows),
            cell=f"gamma={gamma}", label=label,
        )


def validate_evaluations(
    inputs: list[tuple[str, Path]],
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, Any], dict[str, Any]]:
    if not inputs:
        raise ValueError("at least one label=path evaluation is required")
    if len({label for label, _ in inputs}) != len(inputs):
        raise ValueError("evaluation labels must be unique")
    payloads = {label: _load(path) for label, path in inputs}
    baseline_label = inputs[0][0]
    first_per_gamma = payloads[baseline_label].get("summary", {}).get("per_gamma", {})
    if not isinstance(first_per_gamma, dict) or not first_per_gamma:
        raise ValueError("the baseline evaluation has no per-gamma summary")
    gammas = sorted(first_per_gamma, key=float)
    if list(first_per_gamma) != gammas:
        raise ValueError("the baseline per-gamma summary is not numerically ordered")

    source = _source_contract(payloads[baseline_label])
    crn = _crn_contract(payloads[baseline_label])
    for label, _ in inputs:
        payload = payloads[label]
        _validate_payload(payload, label=label, gammas=gammas)
        _require_equal(
            "source contract", _source_contract(payload), source, label=label
        )
        _require_equal("CRN contract", _crn_contract(payload), crn, label=label)
    return payloads, gammas, source, crn


def _table_rows(
    payloads: dict[str, dict[str, Any]],
    gammas: list[str],
    inputs: list[tuple[str, Path]],
) -> list[dict[str, Any]]:
    rows = []
    for label, path in inputs:
        payload = payloads[label]
        cells = [("pooled", None, payload["summary"]["pooled"])]
        cells.extend(
            ("gamma", gamma, payload["summary"]["per_gamma"][gamma])
            for gamma in gammas
        )
        for scope, gamma, cell in cells:
            rows.append({
                "label": label,
                "scope": scope,
                "gamma": gamma,
                "n": int(cell["n"]),
                **{metric: cell.get(metric) for metric in METRICS},
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "input_json": str(path.resolve()),
                "input_sha256": _sha256(path),
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
    payloads: dict[str, dict[str, Any]],
    gammas: list[str],
    labels: list[str],
    output: Path,
) -> None:
    positions = np.arange(len(labels))
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
                _metric_value(
                    payloads[label]["summary"]["per_gamma"][gamma], metric
                )
                for label in labels
            ]
            axis.plot(
                positions, values, color=color, linewidth=1.0, marker="o",
                markersize=3.0, alpha=0.72, label=rf"$\gamma={gamma}$",
            )
        pooled = [
            _metric_value(payloads[label]["summary"]["pooled"], metric)
            for label in labels
        ]
        axis.plot(
            positions, pooled, color="black", linewidth=2.8, marker="o",
            markersize=5.0, label="pooled", zorder=20,
        )
        axis.set_xlabel("Evaluation")
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
        axis.grid(alpha=0.22, linewidth=0.7)
        if metric in {"CR", "Validity"}:
            axis.set_ylim(0.0, 1.0)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.925),
        ncol=min(8, len(legend_labels)), frameon=False, fontsize=8,
    )
    first = payloads[labels[0]]
    scene = first["scene"].get("scene_profile", "unknown")
    figure.suptitle(
        f"SFM2 expansion: fixed raw temperature=1 CRN evaluation\n"
        f"{scene}, M={first['M_per_gamma']}/gamma",
        y=0.995, fontsize=12,
    )
    figure.subplots_adjust(
        left=0.08, right=0.985, bottom=0.12, top=0.84,
        wspace=0.24, hspace=0.38,
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    figure.savefig(temporary, format=output.suffix.lstrip("."), dpi=200)
    plt.close(figure)
    temporary.replace(output)


def build_report(inputs: list[tuple[str, Path]], outdir: Path) -> dict[str, Path]:
    payloads, gammas, source, crn = validate_evaluations(inputs)
    labels = [label for label, _ in inputs]
    rows = _table_rows(payloads, gammas, inputs)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "hp100_expansion_raw_metrics.csv"
    json_path = outdir / "hp100_expansion_raw_metrics.json"
    png_path = outdir / "hp100_expansion_raw_four_metrics.png"
    pdf_path = outdir / "hp100_expansion_raw_four_metrics.pdf"
    _write_csv(csv_path, rows)
    _write_json(json_path, {
        "status": "SFM2_EXPANSION_RAW_REPORT_COMPLETE",
        "labels": labels,
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
    _render(payloads, gammas, labels, png_path)
    _render(payloads, gammas, labels, pdf_path)
    return {"csv": csv_path, "json": json_path, "png": png_path, "pdf": pdf_path}


def _parse_inputs(values: list[str]) -> list[tuple[str, Path]]:
    inputs = []
    for value in values:
        label, _, path = value.partition("=")
        if not label or not path:
            raise ValueError(f"--eval arguments must be label=path, got {value!r}")
        inputs.append((label, Path(path)))
    return inputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval", action="append", required=True,
        help="label=path; repeat per evaluation, first is the baseline",
    )
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args(argv)
    outputs = build_report(_parse_inputs(args.eval), args.outdir)
    print(json.dumps({key: str(path.resolve()) for key, path in outputs.items()}))


if __name__ == "__main__":
    main()
