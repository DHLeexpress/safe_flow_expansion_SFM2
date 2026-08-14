"""Merge several canonical dataset dirs (train/val/test shards) into one, then
re-split. Used to combine the 4 parallel UCY guided-generation shards.

  python -m cfm_mppi.data.merge_canonical --inputs dataset/ucy_guided_g0 \
      dataset/ucy_guided_g1 dataset/ucy_guided_g2 dataset/ucy_guided_g3 \
      --output dataset/ucy_guided_merged
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from cfm_mppi.data.canonical_dataset import save_canonical_splits

TENSOR_KEYS = ["states", "controls_dyn", "controls_si", "start", "goal",
               "ego_history", "action_history", "nearest_obstacle_history",
               "obstacles", "gamma", "safety_margin"]
LIST_KEYS = ["dynamics_type", "source"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", default="dataset/ucy_guided_merged")
    p.add_argument("--seed", type=int, default=0)
    cli = p.parse_args()

    parts = {k: [] for k in TENSOR_KEYS}
    lists = {k: [] for k in LIST_KEYS}
    meta = None
    total = 0
    for d in cli.inputs:
        for split in ("train", "val", "test"):
            fp = Path(d) / f"{split}.pt"
            if not fp.exists():
                continue
            data = torch.load(fp, weights_only=False)
            n = len(data["gamma"])
            total += n
            for k in TENSOR_KEYS:
                parts[k].append(data[k])
            for k in LIST_KEYS:
                v = data.get(k, [None] * n)
                lists[k].extend(list(v) if not isinstance(v, list) else v)
            if meta is None:
                meta = data.get("metadata", {})
    merged = {k: torch.cat(parts[k], dim=0) for k in TENSOR_KEYS}
    merged["dynamics_type"] = lists["dynamics_type"]
    merged["source"] = lists["source"]
    merged["metadata"] = {**(meta or {}), "merged_from": cli.inputs, "merged_total": total}
    paths = save_canonical_splits(merged, Path(cli.output), seed=cli.seed)
    print(f"merged {total} items from {len(cli.inputs)} dirs -> {cli.output}: {paths}")


if __name__ == "__main__":
    main()
