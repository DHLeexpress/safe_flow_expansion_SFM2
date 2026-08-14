"""Build ego+crowd SafeMPPI scenes from the raw ETH/UCY pedestrian data (Pellegrini ICCV 2009 / Lerner 2007).

Unlike dataset/train80_ego.pt (ego-only, NO surroundings), each scene here keeps the EGO pedestrian AND every
co-present pedestrian as a moving obstacle -- exactly what SafeMPPI needs. Raw format: rows of `frame ped x y`
at 2.5 fps (dt_native=0.4 s); we linearly interpolate to dt=0.1 s and cut 8 s (80-step) windows.

Output: dataset/<name>_crowd_scenes.pkl = list of episodes, each:
  { "start":[2], "goal":[2], "obstacles_seq":[T,N,3] (x,y,radius), "velocities_seq":[T,N,2], "source":str }
(N padded with NaN per episode; downstream filters NaN, exactly like eval80_obs).

  python -m cfm_mppi.data.build_crowd_scenes --files biwi_eth biwi_hotel --out dataset/eth_crowd_scenes.pkl
"""
from __future__ import annotations
import argparse, os, pickle
import numpy as np

RAW_DIR = "dataset/eth_ucy_raw"
DT_NATIVE = 0.4
DT_TARGET = 0.1
WIN_T = 8.0
RADIUS = 0.5


def _interp(track_f, track_xy, grid_f):
    """Linear-interp a ped track (sorted native frames) onto grid_f; NaN outside the track's frame range."""
    x = np.interp(grid_f, track_f, track_xy[:, 0])
    y = np.interp(grid_f, track_f, track_xy[:, 1])
    out = np.stack([x, y], 1)
    out[(grid_f < track_f[0]) | (grid_f > track_f[-1])] = np.nan
    return out


def build_episodes(path, win_t=WIN_T, stride_t=4.0, min_track_t=8.0, max_n=24, source="eth"):
    data = np.loadtxt(path)                                    # [R,4]: frame ped x y
    tracks = {}
    for fr, pid, x, y in data:
        tracks.setdefault(int(pid), []).append((fr, x, y))
    for p in tracks:
        tracks[p] = np.array(sorted(tracks[p]))                # [Ti,3]
    uframes = np.array(sorted(set(data[:, 0].tolist())))
    fstep = float(np.min(np.diff(uframes))) if uframes.size > 1 else 10.0   # native frame step (=dt_native)
    T = int(round(win_t / DT_TARGET))                          # 80 steps
    win_fr = win_t / DT_NATIVE * fstep                         # window span in raw-frame units
    stride_fr = stride_t / DT_NATIVE * fstep

    episodes = []
    for ego, tr in tracks.items():
        f0_ego, f1_ego = tr[0, 0], tr[-1, 0]
        if (f1_ego - f0_ego) < (min_track_t / DT_NATIVE) * fstep:
            continue                                            # ego track too short
        start_fr = f0_ego
        while start_fr + win_fr <= f1_ego + 1e-6:
            grid = np.linspace(start_fr, start_fr + win_fr, T)  # [T] raw-frame grid
            ego_xy = _interp(tr[:, 0], tr[:, 1:3], grid)        # [T,2]
            if np.isnan(ego_xy).any():
                start_fr += stride_fr; continue
            # crowd = every OTHER ped overlapping the window
            obs_list = []
            for pid, otr in tracks.items():
                if pid == ego:
                    continue
                if otr[-1, 0] < start_fr or otr[0, 0] > start_fr + win_fr:
                    continue
                oxy = _interp(otr[:, 0], otr[:, 1:3], grid)     # [T,2] (NaN outside its range)
                if np.isnan(oxy).all():
                    continue
                obs_list.append(oxy)
            if len(obs_list) == 0:
                start_fr += stride_fr; continue
            obs = np.stack(obs_list, 1)                          # [T, Nobs, 2]
            N = min(obs.shape[1], max_n)
            obs = obs[:, :N]
            vel = np.zeros_like(obs); vel[1:] = (obs[1:] - obs[:-1]) / DT_TARGET
            ego_vel = np.zeros_like(ego_xy); ego_vel[1:] = (ego_xy[1:] - ego_xy[:-1]) / DT_TARGET
            obstacles_seq = np.concatenate([obs, np.full((T, N, 1), RADIUS)], 2)   # [T,N,3]
            episodes.append({
                "start": ego_xy[0].astype(np.float32),
                "goal": ego_xy[-1].astype(np.float32),
                "obstacles_seq": obstacles_seq.astype(np.float32),
                "velocities_seq": vel.astype(np.float32),
                "ego_seq": np.concatenate([ego_xy, ego_vel], 1).astype(np.float32),
                "source": source,
            })
            start_fr += stride_fr
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=["biwi_eth"],
                    help="raw stems in dataset/eth_ucy_raw (e.g. biwi_eth biwi_hotel crowds_zara01 ...)")
    ap.add_argument("--out", default="dataset/eth_crowd_scenes.pkl")
    ap.add_argument("--stride-t", type=float, default=4.0)
    ap.add_argument("--max-n", type=int, default=24)
    args = ap.parse_args()
    all_eps = []
    for stem in args.files:
        path = os.path.join(RAW_DIR, f"{stem}.txt")
        eps = build_episodes(path, stride_t=args.stride_t, max_n=args.max_n, source=stem)
        ncr = np.mean([(~np.isnan(e["obstacles_seq"][..., 0]).all(0)).sum() for e in eps]) if eps else 0
        print(f"  {stem}: {len(eps)} episodes, mean crowd size {ncr:.1f}", flush=True)
        all_eps += eps
    with open(args.out, "wb") as f:
        pickle.dump(all_eps, f)
    print(f"saved {len(all_eps)} ego+crowd episodes -> {args.out}")


if __name__ == "__main__":
    main()
