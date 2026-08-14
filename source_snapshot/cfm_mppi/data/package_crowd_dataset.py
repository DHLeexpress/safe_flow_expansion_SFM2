"""Clean the ETH/UCY ego+crowd episodes and package them in Mizuta's eval80 format:
   <split>_ego.pt  = tensor [N_ep, 6, 80]            channels = [x, y, vx, vy, sin(heading), cos(heading)]
   <split>_obs.pkl = list[N_ep] of tensor [1, N_ped, 6, 80]   (same 6 channels; NaN where a ped is absent)

Cleaning: (1) clip velocity magnitude to V_MAX (interpolation spikes at track edges); (2) keep only episodes whose
ego actually travels >= MIN_DISP (a real start->goal). Train = UCY students (largest), OOD test = ETH; students is
also split train/val/test so we get an in-distribution test too.

  python -m cfm_mppi.data.package_crowd_dataset
"""
from __future__ import annotations
import argparse, os, pickle
import numpy as np
import torch

V_MAX = 2.5            # m/s; pedestrians rarely exceed this -> clip interpolation glitches
MIN_DISP = 1.5         # m; ego must travel at least this far (real goal)
EPS = 1e-6


def _clip_vel(v):                                   # v [...,2]
    spd = np.linalg.norm(v, axis=-1, keepdims=True)
    scale = np.minimum(1.0, V_MAX / np.clip(spd, EPS, None))
    return v * scale


def _sincos(v):                                     # v [...,2] -> sin,cos of heading
    spd = np.linalg.norm(v, axis=-1)
    s = np.where(spd > EPS, v[..., 1] / np.clip(spd, EPS, None), 0.0)
    c = np.where(spd > EPS, v[..., 0] / np.clip(spd, EPS, None), 1.0)
    return s, c


def _to_mizuta(ep):
    ego_xy = ep["ego_seq"][:, :2]                   # [80,2]
    ego_v = _clip_vel(ep["ego_seq"][:, 2:4])
    es, ec = _sincos(ego_v)
    ego = np.stack([ego_xy[:, 0], ego_xy[:, 1], ego_v[:, 0], ego_v[:, 1], es, ec], 0).astype(np.float32)  # [6,80]
    obs_xy = ep["obstacles_seq"][..., :2]           # [80,N,2]
    obs_v = _clip_vel(ep["velocities_seq"])         # [80,N,2]
    os_, oc = _sincos(obs_v)
    obs = np.stack([obs_xy[..., 0], obs_xy[..., 1], obs_v[..., 0], obs_v[..., 1], os_, oc], -1)            # [80,N,6]
    obs = np.transpose(obs, (1, 2, 0))[None]        # [1,N,6,80]
    return ego, obs.astype(np.float32)


def _save(eps, stem, out_dir):
    egos = []; obss = []
    for ep in eps:
        disp = float(np.linalg.norm(ep["ego_seq"][-1, :2] - ep["ego_seq"][0, :2]))
        if disp < MIN_DISP:
            continue
        ego, obs = _to_mizuta(ep)
        egos.append(torch.from_numpy(ego)); obss.append(torch.from_numpy(obs))
    if not egos:
        print(f"  {stem}: 0 episodes after filter"); return 0
    torch.save(torch.stack(egos), os.path.join(out_dir, f"{stem}_ego.pt"))
    with open(os.path.join(out_dir, f"{stem}_obs.pkl"), "wb") as f:
        pickle.dump(obss, f)
    ncr = np.mean([(~np.isnan(o[0, :, 0, :]).all(1)).sum().item() for o in obss])
    print(f"  {stem}: {len(egos)} episodes (>= {MIN_DISP} m), mean crowd {ncr:.1f}  -> {stem}_ego.pt / {stem}_obs.pkl")
    return len(egos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="dataset/eth_crowd_scenes.pkl")
    ap.add_argument("--out", default="dataset/crowd")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    eps = pickle.load(open(args.pkl, "rb"))
    students = [e for e in eps if e["source"] in ("students001", "students003")]
    eth = [e for e in eps if e["source"] in ("biwi_eth", "biwi_hotel")]
    rng = np.random.default_rng(args.seed); rng.shuffle(students)
    n = len(students); ntr = int(0.8 * n); nva = int(0.1 * n)
    print(f"UCY students: {n} raw  | ETH: {len(eth)} raw  (before MIN_DISP filter)")
    _save(students[:ntr], "students_train", args.out)
    _save(students[ntr:ntr + nva], "students_val", args.out)
    _save(students[ntr + nva:], "students_test", args.out)
    _save(eth, "eth_ood_test", args.out)


if __name__ == "__main__":
    main()
