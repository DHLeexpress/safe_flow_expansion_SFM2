from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List

import numpy as np


class JSONLResultWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, record: Dict) -> None:
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def _p05(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.percentile(values, 5)) if values else float("nan")


def summarize_records(records: Iterable[Dict]) -> List[Dict]:
    groups = defaultdict(list)
    for rec in records:
        key = (rec.get("dataset"), rec.get("dynamics"), rec.get("method"))
        groups[key].append(rec)
    rows = []
    for (dataset, dynamics, method), items in groups.items():
        rows.append(
            {
                "dataset": dataset,
                "dynamics": dynamics,
                "method": method,
                "episodes": len(items),
                "success_rate": _mean([float(x.get("success", False)) for x in items]),
                "collision_rate": _mean([float(x.get("collision", False)) for x in items]),
                "mean_min_clearance": _mean([x.get("min_clearance") for x in items]),
                "median_min_clearance": float(median([x.get("min_clearance") for x in items])) if items else float("nan"),
                "p05_min_clearance": _p05([x.get("min_clearance") for x in items]),
                "mean_final_goal_distance": _mean([x.get("final_goal_distance") for x in items]),
                "mean_control_effort": _mean([x.get("control_effort") for x in items]),
                "mean_control_smoothness": _mean([x.get("control_smoothness") for x in items]),
                "mean_planning_time_ms": 1000.0 * _mean([x.get("planning_wall_time_mean") for x in items]),
                "p95_planning_time_ms": 1000.0 * _mean([x.get("planning_wall_time_p95") for x in items]),
                "mean_nfe": _mean([x.get("nfe") for x in items]),
            }
        )
    return rows


def write_summary(output_dir: str | Path, records: List[Dict]) -> Dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = summarize_records(records)
    csv_path = out / "summary.csv"
    json_path = out / "summary.json"
    md_path = out / "summary.md"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = ["# Benchmark Summary", ""]
    for row in rows:
        lines.append(
            f"- {row['dataset']} / {row['dynamics']} / {row['method']}: "
            f"success={row['success_rate']:.3f}, collision={row['collision_rate']:.3f}, "
            f"mean_min_clearance={row['mean_min_clearance']:.3f}, episodes={row['episodes']}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"summary_csv": csv_path, "summary_json": json_path, "summary_md": md_path}
