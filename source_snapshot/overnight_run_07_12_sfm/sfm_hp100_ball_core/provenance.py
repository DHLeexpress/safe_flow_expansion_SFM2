"""Authentication for the 3-D-derived, explicitly patched SFM core.

The original source checkout was dirty when copied, so its content hash remains
the authoritative base identity. The SFM semantic delta and patched-port hash
are authenticated separately; this module never claims byte identity after the
documented SFM-specific archive/GP change.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


SOURCE_REPOSITORY = "/Users/dhl/Documents/safeMPPI_demo_3d"
SOURCE_RELATIVE_PATH = "safe_mppi/expansion.py"
SOURCE_HEAD = "5c8a57779f165c583b297b73ab6d8bf90e3f59f5"
SOURCE_WORKTREE_WAS_DIRTY = True
SOURCE_SHA256 = "b1b50812a3d409646b7220c805ed787b4ae2da3fdf5ac7ec39cfc629bc3e4d68"
PATCHED_PORT_SHA256 = "0f039757e85bc75524e398a8f3bf979bcd573f687f06c27ed088ce485f567cf4"
SEMANTIC_DELTA_FILE = "SEMANTIC_DELTA.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_vendored_core() -> dict:
    path = Path(__file__).with_name("expansion.py")
    actual = _sha256(path)
    if actual != PATCHED_PORT_SHA256:
        raise RuntimeError(
            "patched SFM expansion core hash mismatch: "
            f"{actual} != {PATCHED_PORT_SHA256}"
        )
    delta_path = Path(__file__).with_name(SEMANTIC_DELTA_FILE)
    if not delta_path.is_file():
        raise RuntimeError("SFM expansion semantic-delta manifest is missing")
    return {
        "vendored_path": str(path.resolve()),
        "sha256": actual,
        "patched_port_sha256": actual,
        "base_source_sha256": SOURCE_SHA256,
        "source_repository": SOURCE_REPOSITORY,
        "source_relative_path": SOURCE_RELATIVE_PATH,
        "source_head": SOURCE_HEAD,
        "source_worktree_was_dirty": SOURCE_WORKTREE_WAS_DIRTY,
        "semantic_delta_manifest": str(delta_path.resolve()),
        "semantic_delta_manifest_sha256": _sha256(delta_path),
        "identity_note": (
            "authenticated SFM-specific patched port derived from the recorded "
            "3-D source content; see semantic_delta_manifest for every declared "
            "archive/GP semantic change"
        ),
    }
