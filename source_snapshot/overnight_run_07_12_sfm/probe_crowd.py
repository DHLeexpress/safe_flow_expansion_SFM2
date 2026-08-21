"""Quantify the frozen crowd's parking behaviour versus the re-targeting fork.

CPU-only probe: replay the scene's pedestrians for one episode with a synthetic
robot walking start->goal, under the frozen dynamics and under
``ped_extend.retargeting_pedestrians``.  Reports how many pedestrians are still
moving over time and how close any pedestrian gets to the robot goal (the
video must not have the crowd parking on the goal).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import _paths  # noqa: F401
import sfm_scene as SS
import ped_extend


def roll(episode: int, steps: int, patched: bool) -> dict:
    environment = SS.scene_profile("double_density_velocity_ood")
    humans = SS.make_humans(
        int(episode), seed=0, n_ped=int(environment["n_ped"]),
        speed_range=tuple(environment["ped_speed_range"]),
    )
    robot = np.array([0.0, 0.0, 0.0, 0.0], np.float32)
    direction = (SS.GOAL - robot[:2]) / np.linalg.norm(SS.GOAL - robot[:2])
    moving, goal_gap, previous = [], [], SS.collect_humans(humans)[0].copy()
    context = (ped_extend.retargeting_pedestrians() if patched
               else _null_context())
    with context:
        for step in range(steps):
            speed = 1.0 if np.linalg.norm(robot[:2] - SS.GOAL) > 0.3 else 0.0
            robot[2:4] = direction * speed
            robot[:2] = robot[:2] + robot[2:4] * SS.DT
            SS.advance_humans(humans, robot)
            xy, _vel = SS.collect_humans(humans)
            moving.append(float((np.linalg.norm(xy - previous, axis=1)
                                 > 1.0e-4).mean()))
            goal_gap.append(float(np.linalg.norm(xy - SS.GOAL, axis=1).min()))
            previous = xy.copy()
    window = max(1, steps // 6)
    return {
        "moving_fraction_by_sixth": [
            round(float(np.mean(moving[i:i + window])), 3)
            for i in range(0, steps, window)
        ],
        "moving_fraction_final_20": round(float(np.mean(moving[-20:])), 3),
        "min_pedestrian_distance_to_goal": round(float(np.min(goal_gap)), 3),
        "final_min_distance_to_goal": round(float(goal_gap[-1]), 3),
    }


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=920036)
    ap.add_argument("--steps", type=int, default=180)
    args = ap.parse_args()
    report = {
        "episode": args.episode, "steps": args.steps,
        "frozen": roll(args.episode, args.steps, patched=False),
        "retargeting": roll(args.episode, args.steps, patched=True),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
