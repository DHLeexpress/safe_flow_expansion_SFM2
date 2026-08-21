"""Video-only pedestrian re-targeting variant of the frozen SFM crowd.

The frozen ``HumanAgent.social_force_step`` parks an agent forever once it is
within 0.1 m of its sampled goal::

    if np.linalg.norm(self.goal - self.state) < 0.1:
        self.control = np.zeros(2); return

Goals are drawn uniformly in [-2, 8]^2 and double-OOD speeds are 1.0-2.0 m/s,
so most of the 40 pedestrians arrive within 30-60 steps and the late frames of
a long episode show a robot moving through a parked crowd.  For presentation
videos we want the crowd to stay alive for the whole episode, so this module
installs a scoped patch that draws a fresh goal on arrival instead of freezing.

Contract of the patch:

* the frozen module is never edited — ``HumanAgent.social_force_step`` is
  rebound inside a context manager and restored on exit (same discipline as
  the acquisition selector patch);
* the re-drawn goal reuses the constructor's rules — uniform in [-2, 8]^2 and
  at least ``GOAL_KEEPOUT`` metres away from the robot goal (6, 6), so no
  pedestrian ever parks on or loiters at the robot's goal — plus a minimum
  travel distance so an agent cannot re-arrive on the same frame;
* the agent's own ``rng`` supplies the draw, so a scenario id still fixes the
  whole crowd trajectory and the variant is reproducible;
* this changes the simulated crowd, therefore rollouts under the patch are
  **not** the official fixed-bank evaluation and must never be reported as a
  metric — they exist to make the comparison video legible.
"""
from __future__ import annotations

import contextlib

import numpy as np

import _paths  # noqa: F401
import sfm_scene as SS
from cfm_mppi.utils import HumanAgent

ARRIVAL_RADIUS = 0.1      # the frozen arrival test we are replacing
GOAL_KEEPOUT = 2.0        # HumanAgent.__init__ keep-out from the robot goal
TRAVEL_MIN = 1.5          # a fresh goal must be a real walk away
BOUND_LO, BOUND_HI = -2.0, 8.0
MAX_DRAWS = 64


def _fresh_goal(agent) -> np.ndarray | None:
    for _ in range(MAX_DRAWS):
        candidate = agent.rng.uniform(BOUND_LO, BOUND_HI, size=(2,))
        if np.linalg.norm(candidate - SS.GOAL) < GOAL_KEEPOUT:
            continue
        if np.linalg.norm(candidate - agent.state) < TRAVEL_MIN:
            continue
        return candidate
    return None


@contextlib.contextmanager
def retargeting_pedestrians():
    """Scoped: pedestrians re-target on arrival instead of parking forever."""
    original = HumanAgent.social_force_step

    def patched(self, others_states, others_controls, tau=0.5):
        if np.linalg.norm(self.goal - self.state) < ARRIVAL_RADIUS:
            candidate = _fresh_goal(self)
            if candidate is not None:
                self.goal = candidate
        return original(self, others_states, others_controls, tau=tau)

    HumanAgent.social_force_step = patched
    try:
        yield
    finally:
        HumanAgent.social_force_step = original


def crowd_motion_profile(ped_xy_by_step) -> dict:
    """Per-step share of pedestrians that actually moved (audit statistic)."""
    frames = [np.asarray(frame, float) for frame in ped_xy_by_step]
    moving = []
    for previous, current in zip(frames[:-1], frames[1:]):
        step = np.linalg.norm(current - previous, axis=1)
        moving.append(float((step > 1.0e-4).mean()))
    return {
        "steps": len(moving),
        "moving_fraction_first": moving[0] if moving else None,
        "moving_fraction_last": moving[-1] if moving else None,
        "moving_fraction_mean": float(np.mean(moving)) if moving else None,
    }
