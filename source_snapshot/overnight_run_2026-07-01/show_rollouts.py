"""Quick look: SafeMPPI di_grid rollouts (black-dot trail) for BOTH scenes, before the big pipeline run."""
from __future__ import annotations

import _paths
import config as C
import di_grid_viz as V


def main():
    cfg = V.load_best_config()
    for scene in C.SCENE_NAMES:
        env = C.make_scene(scene, T=60)
        data = {g: V.mppi_rollout(env, g, cfg, steps=60, seed_base=int(g * 1000)) for g in C.GAMMAS}
        out = C.scene_fig(scene, f"safemppi_{scene}")
        V.render_grid(env, data, C.GAMMAS, out, polytope_mode="nominal",
                      title=f"SafeMPPI  [{scene}]  γ-sweep (blue = nominal polytope, black = executed states)",
                      mp4=True)
        print(f"[{scene}] rendered", flush=True)


if __name__ == "__main__":
    main()
