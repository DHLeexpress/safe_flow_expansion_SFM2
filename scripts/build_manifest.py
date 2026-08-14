#!/usr/bin/env python3
"""Build the byte-level package inventory from files tracked by Git."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SOURCE_MANIFEST.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tracked_files() -> list[Path]:
    """Return the committed package surface, excluding this derived manifest."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    relative_paths = [
        Path(raw.decode())
        for raw in result.stdout.split(b"\0")
        if raw and raw.decode() != OUTPUT.name
    ]
    missing = [path.as_posix() for path in relative_paths if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"tracked package files are missing: {missing}")
    return sorted(relative_paths)


def main() -> None:
    rows = []
    for relative in tracked_files():
        path = ROOT / relative
        rows.append({
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        })
    payload = {
        "status": "SAFE_FLOW_EXPANSION_SFM_PACKAGE",
        "source_lineage": (
            "canonical HP100: dataset safeMPPI@2671a9447b7b914053dce5fe9be2a0aae6c67a8d"
            " + pretraining safeMPPI@e9164e5a6e70b86cecae4660e7732f8ecc6a93f7"
            " + branch renderer safeMPPI@b659526"
            " + integrated snapshot safeMPPI@ef35f1f2df89a2131d9e0a21e0d7095a2d1d7b1d"
        ),
        "files": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUTPUT} with {len(rows)} files")


if __name__ == "__main__":
    main()
