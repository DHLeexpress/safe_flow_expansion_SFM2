from __future__ import annotations

import argparse
from pathlib import Path

from .canonical_dataset import (
    build_canonical_from_mizuta,
    build_canonical_from_safegpc,
    describe_canonical,
    save_canonical_splits,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic canonical benchmark dataset splits.")
    parser.add_argument("--source", choices=["mizuta", "safeGPC"], default="mizuta")
    parser.add_argument("--input", required=True, help="Input .pt/.pkl/.jsonl/.csv/.npz file or directory.")
    parser.add_argument("--output-dir", default="dataset/canonical")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history-len", type=int, default=None)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    if args.source == "mizuta":
        data = build_canonical_from_mizuta(
            args.input,
            history_len=args.history_len or 10,
            dt=args.dt,
            max_items=args.max_items,
        )
    else:
        data = build_canonical_from_safegpc(
            args.input,
            history_len=args.history_len or 11,
            dt=args.dt,
            max_episodes=args.max_episodes,
        )

    paths = save_canonical_splits(data, Path(args.output_dir), seed=args.seed)
    print(describe_canonical(data))
    print("saved:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
