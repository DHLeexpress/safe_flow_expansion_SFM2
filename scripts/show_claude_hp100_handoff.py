#!/usr/bin/env python3
"""Verify and print the immutable HP100 Claude-handoff baseline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "checkpoint": (
        "checkpoints/hp100_pretrained_r0_258999ae.pt",
        "258999ae8ccee8aec5aab92a6f751221d3c15583ac26e0a7ec8311f13316ec44",
    ),
    "pretraining_report": (
        "provenance/hp100_pretrain_20260802/pretraining_report.json",
        "19f83f865056db73aea656d617680c1078ed5cf69e0a5fdb68cc4f5c14b74cbd",
    ),
    "dataset_manifest": (
        "provenance/hp100_pretrain_20260802/dataset_manifest.json",
        "44f2bfa8afbb2318376ae9e188b1b622f102253a4a91f5c8ca0f9634d5041c94",
    ),
    "id_m50": (
        "provenance/hp100_pretrain_20260802/id_m50.json",
        "58df71d0a25b801c47f9f0da8077704eddb0b52eda18c072de45cad2c3818961",
    ),
    "ood_m50": (
        "provenance/hp100_pretrain_20260802/ood_m50.json",
        "1708348be707868d93e8e878f069cb66c0605fea33ab1f4319dd3d3cf1b0ce4e",
    ),
    "branch_video": (
        "assets/hp100_20260802/branch_viz/hp100_id_ood_branch.mp4",
        "dd1269a392b3dd14161d2392ad4ee69d278c3bd76253639dac892206370c8547",
    ),
    "expert_video": (
        "assets/hp100_20260802/expert_mechanism/hp100_safemppi_mechanism_g0p2_ep109.mp4",
        "416b77aeb60b36a06e365dcbb06a343b038bea3a4e671d4c4df00b0f74f69876",
    ),
    "raw_after_template": (
        "assets/templates/raw_after_template.mp4",
        "9d6d399cb84e0c7dc36fda384f6885d72d8ebb4b50f7624e059587aaa83421cf",
    ),
    "acquisition_template": (
        "assets/templates/acquisition_before_template.mp4",
        "032ba56e69d8f1eedd705d7546ad85378f6f779db939dd5a8f504af38a29e4ba",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pooled(path: Path) -> dict:
    payload = json.loads(path.read_text())
    return payload["summary"]["pooled"]


def main() -> None:
    print("HP100 CLAUDE HANDOFF: authenticated local bundle")
    for name, (relative, expected) in EXPECTED.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"{name}: missing {path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"{name}: {actual} != {expected}")
        print(f"PASS {name:20s} {actual}  {relative}")

    for label, key in (("ID", "id_m50"), ("OOD", "ood_m50")):
        row = pooled(ROOT / EXPECTED[key][0])
        print(
            f"{label:3s} raw M50  SR={row['SR']:.4f} CR={row['CR']:.4f} "
            f"TO={row['timeout']:.4f} Validity={row['Validity']:.4f} "
            f"clearance={row['successful_clearance']:.4f}m "
            f"time={row['successful_time_to_goal']:.3f}s"
        )

    print("Checkpoint: canonical HP100 r0; raw temperature=1, NFE=8")
    print("Dataset: 3,500 successful ID lineages; OOD not used for promotion")
    print("Acquisition: always-on K=64/B=32, normalized ESS target=0.1")
    print("Execution: exact-positive lexicographic H10 progress/clearance/sigma")
    print("Replay target: L_positive - alpha*L_negative; no neutral/teacher stream")
    print("Optimizer: flow trunk input + both residual blocks + head; encoders frozen")
    print("Next: read CLAUDE_HP100_EXPANSION_HANDOFF.md completely")


if __name__ == "__main__":
    main()
