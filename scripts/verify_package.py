#!/usr/bin/env python3
"""Verify every file declared by SOURCE_MANIFEST.json."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    failures = []
    for row in manifest["files"]:
        path = ROOT / row["path"]
        if not path.is_file():
            failures.append(f"missing: {row['path']}")
            continue
        if path.stat().st_size != row["bytes"]:
            failures.append(f"size: {row['path']}")
        elif digest(path) != row["sha256"]:
            failures.append(f"sha256: {row['path']}")
    if failures:
        raise SystemExit("package verification failed:\n" + "\n".join(failures))
    print(f"verified {len(manifest['files'])} files")


if __name__ == "__main__":
    main()

