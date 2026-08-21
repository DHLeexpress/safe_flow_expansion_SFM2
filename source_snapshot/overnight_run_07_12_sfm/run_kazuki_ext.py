"""Kazuki CFM-MPPI rollouts under the video-only re-targeting crowd."""
import _paths  # noqa: F401
import kazuki_run
import ped_extend

if __name__ == "__main__":
    with ped_extend.retargeting_pedestrians():
        raise SystemExit(kazuki_run.main())
