"""Run Safe Flow Expansion for MANY rounds — iterate UNTIL all scene modes are discovered AND the certified
coverage near the obstacles saturates (non-negotiable: keep going, do not stop early). Record EVERY round:
certified trajectories (+modes), the live kernel matrix K=<phi_s,phi_s'>, the sigma histogram (GP over phi_s),
coverage, and new-mode events. Saves results/<scene>/movie_record.pkl for the 2x2 movie renderer.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np
import torch

import _paths
import config as C
import coverage as COV
import validity as VAL
from windowed_policy import GridLowFlowPolicy, fm_rollout
from expansion import finetune, collect_fm_positives, collect_broad_positives, load_pretrained
from polar_grid import polar_grid
from local_frame import low_dim_features, goal_frame, to_world
from di_grid_viz import di_step
from uncertainty import GPUncertainty
from cfm_mppi.data.windowed_dataset import WindowedDataset

BETA = 0.077


def probe_state(env):
    obs = env.obstacles.detach().cpu().numpy()
    return np.array([float(obs[:, 0].min()) - 1.0, 0.0, 1.2, 0.0], np.float32)


@torch.no_grad()
def kernel_and_sigma(pol, env, gamma, device, n=90, s=0.9):
    goal = env.goal.detach().cpu().numpy(); obs = env.obstacles.detach().cpu().numpy()
    state = probe_state(env)
    grid, _ = polar_grid(state[:2], goal, obs, r_robot=float(env.r_robot))
    low, _ = low_dim_features(state, goal, gamma, a_prev=np.array([1.2, 0]), prev_valid=True)
    ctx = pol.ctx_from(torch.tensor(grid[None], device=device), torch.tensor(low[None], device=device))
    buf = pol.sample(48, ctx.expand(48, -1), temp=0.7)
    U = pol.sample(n, ctx.expand(n, -1), temp=1.6)
    phi_buf = pol.phi_s(buf, ctx.expand(48, -1), s=s)
    phi = pol.phi_s(U, ctx.expand(n, -1), s=s)
    unc = GPUncertainty(kernel="linear", lam=1e-2, normalize=True); unc.set_buffer(phi_buf)
    sigma = unc.sigma(phi).detach().cpu().numpy()
    phn = torch.nn.functional.normalize(phi, dim=1)
    K = (phn @ phn.T).detach().cpu().numpy()
    e_g, e_lat, _ = goal_frame(state[:2], goal)
    lat = []
    for Ul in U.detach().cpu().numpy():
        Uw = to_world(Ul, e_g, e_lat); s2 = state.copy()
        for u in Uw:
            s2 = di_step(s2, np.clip(u, -env.u_max, env.u_max), dt=env.dt)
        lat.append(float(s2[1]))
    order = np.argsort(lat)                                # sort candidates by window direction (up/down)
    return K[order][:, order], sigma, np.array(lat)[order]


def _ess(sigma):
    w = np.exp((sigma - sigma.max()) / BETA); w /= w.sum()
    return float(1.0 / (w ** 2).sum())


def run(scene, max_rounds, device, cov_goal=0.55, log=print):
    env = C.make_scene(scene); pol = load_pretrained(scene, device)
    demo = WindowedDataset(os.path.join(C.dataset_dir(scene), "train.pt"))
    star = COV.build_omega_star(env, C.VERIFIER["gamma_max"], n=1500)
    star_cells = set(star["cells"]); all_modes = set(star["modes"])
    pos = {"grid": [], "low": [], "U": [], "mode": []}
    for g in C.GAMMAS:
        bg, bl, bu, bm = collect_broad_positives(env, g, 200, seed=int(g * 7), H_pred=C.H_PRED, device=device)
        pos["grid"] += bg; pos["low"] += bl; pos["U"] += bu; pos["mode"] += bm
    acc_cells, seen_modes, records = set(), set(), []
    t0 = time.time()
    for r in range(1, max_rounds + 1):
        for g in C.GAMMAS:
            _, pg, pl, pu, pm = collect_fm_positives(pol, env, g, 24, temp=1.6, device=device)
            pos["grid"] += pg; pos["low"] += pl; pos["U"] += pu; pos["mode"] += pm
            bg, bl, bu, bm = collect_broad_positives(env, g, 140, seed=r * 100 + int(g * 10),
                                                     H_pred=C.H_PRED, device=device)
            pos["grid"] += bg; pos["low"] += bl; pos["U"] += bu; pos["mode"] += bm
        finetune(pol, demo, pos, 200, batch=128, lr=2e-4, device=device)
        round_trajs = []
        for g in C.GAMMAS:
            paths, _ = fm_rollout(pol, env, g, n_traj=24, temp=1.15, device=device, record=False)
            for p in paths:
                if VAL.is_valid(p, env, g):
                    m = COV.mode_of(p, env)
                    round_trajs.append((p.astype(np.float32), m, float(g)))
                    acc_cells |= (COV.spatial_cells([p], env) & star_cells)
        mc = {}
        for (_, m, _) in round_trajs:
            mc[m] = mc.get(m, 0) + 1
        round_modes = {m for m, cnt in mc.items() if cnt >= 2}     # "discovered" = reliably generated (>=2x)
        newly = sorted(round_modes - seen_modes); seen_modes |= round_modes
        cov = len(acc_cells) / max(len(star_cells), 1)
        K, sigma, lat = kernel_and_sigma(pol, env, C.GAMMAS[1], device)
        records.append({"round": r, "trajs": round_trajs, "cov": cov, "ncells": len(acc_cells),
                        "modes": sorted(seen_modes), "new": newly, "K": K, "sigma": sigma,
                        "lat": lat, "ess": _ess(sigma)})
        log(f"[r{r}] cov {cov:.2f} modes {sorted(seen_modes)} NEW {newly} ESS {_ess(sigma):.0f}/{len(sigma)} "
            f"({time.time()-t0:.0f}s)")
        if seen_modes >= all_modes and cov >= cov_goal and r >= 8:
            log(f"[GOAL REACHED] all modes {sorted(all_modes)} + cov {cov:.2f} at round {r}")
            break
    else:
        log(f"[max rounds] stopped at {max_rounds}; modes {sorted(seen_modes)}/{sorted(all_modes)}")
    out = C.scene_result(scene, "movie_record.pkl")
    with open(out, "wb") as f:
        pickle.dump({"scene": scene, "star_cells_n": len(star_cells), "all_modes": sorted(all_modes),
                     "obs": env.obstacles.detach().cpu().numpy(), "goal": env.goal.detach().cpu().numpy(),
                     "x0": env.x0.detach().cpu().numpy(), "xlim": list(env.xlim), "ylim": list(env.ylim),
                     "records": records}, f)
    torch.save({"state_dict": pol.state_dict(), "H_pred": C.H_PRED, "u_max": float(env.u_max),
                "scene": scene}, C.scene_result(scene, "expanded.pt"))
    log(f"saved {out} ({len(records)} rounds) + expanded.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="gap", choices=C.SCENE_NAMES)
    ap.add_argument("--max-rounds", type=int, default=40)
    ap.add_argument("--cov-goal", type=float, default=0.55)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    print(f"=== EXPANSION RECORD [{args.scene}] iterate until all modes + cov>={args.cov_goal} "
          f"(max {args.max_rounds}) ===", flush=True)
    run(args.scene, args.max_rounds, args.device, cov_goal=args.cov_goal)


if __name__ == "__main__":
    main()
