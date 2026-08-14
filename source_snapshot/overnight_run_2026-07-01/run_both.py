"""Orchestrator — run the whole windowed pipeline (dataset → pretrain → expand) for BOTH scenes.

    python run_both.py                       # full: gap + slalom, W&B online
    python run_both.py --smoke               # fast end-to-end check
    python run_both.py --scenes gap          # one scene
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, *args):
    cmd = [sys.executable, os.path.join(HERE, script), *[str(a) for a in args]]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["gap", "slalom"])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    args = ap.parse_args()
    smoke = ["--smoke"] if args.smoke else []
    wb = ["--wandb-mode", args.wandb_mode]

    for scene in args.scenes:
        print(f"\n############### SCENE: {scene} ###############", flush=True)
        run("stage2_build_dataset.py", "--scene", scene, "--episodes", args.episodes, *smoke, *wb)
        run("stage3_pretrain.py", "--scene", scene, "--epochs", args.epochs, "--device", args.device, *smoke, *wb)
        run("expansion.py", "--scene", scene, "--rounds", args.rounds, "--device", args.device, *smoke, *wb)
    print("\n=== BOTH pipelines complete — see figures/<scene>/ and results/<scene>/ ===", flush=True)


if __name__ == "__main__":
    main()
