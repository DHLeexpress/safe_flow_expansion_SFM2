from __future__ import annotations
import csv, json, math, time, zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUT = Path('/mnt/data/pillar3_m_bounds_final')
OUT.mkdir(parents=True, exist_ok=True)

R = 2.0
H = 10
RHO_ART = 0.16
REAL_OBS = [(0.78, 0.55, 0.35), (0.78, -0.55, 0.35)]
t = np.arange(H + 1, dtype=float)
TRAJ = np.stack([1.28 * t / H, 0.035 * np.sin(np.pi * t / H)], axis=1)
TRAJ[0] = 0.0
GAMMAS = [0.3, 0.5, 0.8]
K_VALUES = [4, 8, 16]
# 6 rows = each K shown twice, with loose and tight upper margin caps.
ROW_MODES = [
    ('loose m-bounds', 0.03, 5.00),
    ('tight m-bounds', 0.03, 0.40),
]
N_THETA = 6000

@dataclass
class Face:
    a: np.ndarray
    m: float
    kind: str
    label: str
    ok: bool
    theta: float = 0.0
    reason: str = 'ok'
    ub_obst: float = 0.0
    lb_traj: float = 0.0


def artificial_obstacles(K: int, rho: float = RHO_ART):
    M = R * math.cos(math.pi / K)
    out = []
    for ell in range(K):
        th = 2.0 * math.pi * ell / K
        n = np.array([math.cos(th), math.sin(th)])
        center = (M + rho) * n
        out.append((float(center[0]), float(center[1]), rho))
    return out


def solve_face_bounded_margin(d: np.ndarray, radius: float, traj: np.ndarray, beta: np.ndarray,
                              m_min: float, m_max: float, kind: str, label: str) -> Face:
    d = np.asarray(d, dtype=float).reshape(2)
    pts = traj[1:]
    bts = beta[1:]
    ths = np.linspace(-math.pi, math.pi, N_THETA, endpoint=False)
    best = None
    for th in ths:
        a = np.array([math.cos(th), math.sin(th)], dtype=float)
        ub_obst = float(a @ d - radius)
        # Need positive obstacle margin at least m_min.
        ub = min(m_max, ub_obst)
        # For beta_t > 0, trajectory imposes lower bounds on m.
        vals = (pts @ a) / bts
        lb_traj = float(np.max(vals))
        lb = max(m_min, lb_traj)
        if lb <= ub + 1e-8:
            m = ub  # maximize margin subject to the cap.
            # prefer larger m, then closer to forward direction of obstacle.
            score = (m, a @ d)
            if best is None or score > best[0]:
                best = (score, Face(a=a, m=m, kind=kind, label=label, ok=True,
                                    theta=float(th), ub_obst=ub_obst, lb_traj=lb_traj))
    if best is None:
        return Face(np.array([1.0, 0.0]), 0.0, kind, label, False, reason='infeasible_under_m_bounds')
    face = best[1]
    return face


def polygon(U: np.ndarray, pad: float = 8.0):
    poly = np.array([[-pad, -pad], [pad, -pad], [pad, pad], [-pad, pad]], dtype=float)
    for u in U:
        poly = clip_poly(poly, u, 1.0)
        if len(poly) == 0:
            break
    if len(poly) > 2:
        ctr = poly.mean(axis=0)
        order = np.argsort(np.arctan2(poly[:, 1] - ctr[1], poly[:, 0] - ctr[0]))
        poly = poly[order]
    return poly


def clip_poly(poly: np.ndarray, a: np.ndarray, b: float, tol: float = 1e-10):
    if len(poly) == 0:
        return poly
    out = []
    prev = poly[-1]
    pv = float(a @ prev - b)
    pins = pv <= tol
    for cur in poly:
        cv = float(a @ cur - b)
        cins = cv <= tol
        if cins != pins:
            den = float(a @ (cur - prev))
            if abs(den) > 1e-12:
                lam = float((b - a @ prev) / den)
                out.append(prev + lam * (cur - prev))
        if cins:
            out.append(cur)
        prev, pins = cur, cins
    return np.asarray(out, dtype=float)


def poly_area(poly: np.ndarray):
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def H_grid(U: np.ndarray, GX: np.ndarray, GY: np.ndarray):
    if len(U) == 0:
        return np.full_like(GX, -1.0)
    pts = np.c_[GX.ravel(), GY.ravel()]
    return (1.0 - pts @ U.T).min(axis=1).reshape(GX.shape)


def levelset_ok(U: np.ndarray, traj: np.ndarray, alpha: np.ndarray):
    if len(U) == 0:
        return False, -float('inf'), -1, []
    Hvals = (1.0 - traj @ U.T).min(axis=1)
    margins = Hvals - alpha
    return bool(np.min(margins[1:]) >= -2e-6), float(np.min(margins[1:])), int(np.argmin(margins[1:]) + 1), Hvals.tolist()


def build_solution(K: int, gamma: float, m_min: float, m_max: float):
    alpha = (1.0 - gamma) ** t
    beta = 1.0 - alpha
    faces = []
    for j, (ox, oy, rr) in enumerate(REAL_OBS):
        faces.append(solve_face_bounded_margin(np.array([ox, oy]), rr, TRAJ, beta, m_min, m_max, 'real', f'real{j}'))
    art = artificial_obstacles(K)
    for ell, (ox, oy, rr) in enumerate(art):
        faces.append(solve_face_bounded_margin(np.array([ox, oy]), rr, TRAJ, beta, m_min, m_max, 'artificial', f'art{ell}'))
    ok_faces = [f for f in faces if f.ok and f.m > 1e-8]
    U = np.vstack([f.a / f.m for f in ok_faces]) if ok_faces else np.zeros((0, 2))
    certified, min_margin, worst_t, Hvals = levelset_ok(U, TRAJ, alpha)
    poly = polygon(U, pad=8.0)
    area = poly_area(poly)
    real_ms = [f.m for f in faces if f.kind == 'real' and f.ok]
    art_ms = [f.m for f in faces if f.kind == 'artificial' and f.ok]
    return faces, U, art, {
        'K': K, 'gamma': gamma, 'm_min': m_min, 'm_max': m_max, 'certified': bool(certified),
        'min_levelset_margin': float(min_margin), 'worst_t': int(worst_t), 'area': float(area),
        'n_faces_total': int(len(faces)), 'n_faces_ok': int(len(ok_faces)), 'all_faces_ok': bool(all(f.ok for f in faces)),
        'real_margin_mean': float(np.mean(real_ms)) if real_ms else float('nan'),
        'art_margin_mean': float(np.mean(art_ms)) if art_ms else float('nan'),
        'Hvals': Hvals,
    }


def draw_panel(ax, U, art, K, gamma, mode_label, metrics):
    xlim = (-2.7, 2.7)
    ylim = (-2.35, 2.35)
    gx = np.linspace(*xlim, 340)
    gy = np.linspace(*ylim, 300)
    GX, GY = np.meshgrid(gx, gy)
    Hh = H_grid(U, GX, GY)
    alpha = (1.0 - gamma) ** t
    levels = sorted(set([0.0, 1.0001] + [round(float(alpha[i]), 6) for i in [1, 2, 4, 7, 10] if 0 < alpha[i] < 1]))
    if len(levels) >= 2:
        ax.contourf(GX, GY, Hh, levels=levels, cmap='Greens', alpha=0.33, zorder=1)
    ax.contour(GX, GY, Hh, levels=[0.0], colors='#006d2c', linewidths=1.7, zorder=5)
    inner = [v for v in levels if 0 < v < 1]
    if inner:
        ax.contour(GX, GY, Hh, levels=inner, colors='#238b45', linewidths=0.35, alpha=0.8, zorder=4)
    poly = polygon(U, pad=8.0)
    if len(poly) >= 3:
        ax.plot(np.r_[poly[:,0], poly[0,0]], np.r_[poly[:,1], poly[0,1]], color='#006d2c', lw=1.0, zorder=6)
    ax.add_patch(Circle((0,0), R, facecolor='none', edgecolor='0.55', lw=0.9, ls=':', zorder=2))
    for ox, oy, rr in art:
        ax.add_patch(Circle((ox, oy), rr, facecolor='none', edgecolor='0.20', lw=0.7, ls='--', alpha=0.75, zorder=3))
        ax.plot(ox, oy, marker='x', color='0.15', ms=3.0, mew=0.75, zorder=4)
    for ox, oy, rr in REAL_OBS:
        ax.add_patch(Circle((ox,oy), rr, facecolor='#c8a2c8', edgecolor='#7b3294', lw=1.0, alpha=0.82, zorder=8))
    ax.plot(TRAJ[:,0], TRAJ[:,1], 'k.-', lw=1.35, ms=3.8, zorder=10)
    ax.scatter([TRAJ[0,0]], [TRAJ[0,1]], s=36, c='#00a000', edgecolor='k', zorder=11)
    text = (f'{mode_label}\nK={K}, γ={gamma}\n'
            f'm∈[{metrics["m_min"]:.2f}, {metrics["m_max"]:.2f}]\n'
            f'cert={metrics["certified"]}, area={metrics["area"]:.3f}\n'
            f'mR={metrics["real_margin_mean"]:.3f}, mA={metrics["art_margin_mean"]:.3f}')
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va='top', fontsize=6.8,
            bbox=dict(boxstyle='round', fc='white', alpha=0.88, lw=0.25), zorder=20)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])


def main():
    rows = []
    cache = {}
    for K in K_VALUES:
        for mode_label, m_min, m_max in ROW_MODES:
            for gamma in GAMMAS:
                for _ in range(1):
                    build_solution(K, gamma, m_min, m_max)
                nrep = 3
                tic = time.perf_counter()
                out = None
                for _ in range(nrep):
                    out = build_solution(K, gamma, m_min, m_max)
                elapsed = 1000.0 * (time.perf_counter() - tic) / nrep
                faces, U, art, metrics = out
                metrics['solve_time_ms'] = elapsed
                metrics['mode'] = mode_label
                rows.append(metrics)
                cache[(K, mode_label, gamma)] = (faces, U, art, metrics)

    fig, axes = plt.subplots(6, 3, figsize=(13.4, 23.2), squeeze=False)
    r = 0
    for K in K_VALUES:
        for mode_label, m_min, m_max in ROW_MODES:
            for ci, gamma in enumerate(GAMMAS):
                faces, U, art, metrics = cache[(K, mode_label, gamma)]
                draw_panel(axes[r, ci], U, art, K, gamma, mode_label, metrics)
                if r == 0:
                    axes[r, ci].set_title(f'γ={gamma}', fontsize=11)
                if ci == 0:
                    axes[r, ci].set_ylabel(f'K={K}\n{mode_label}', fontsize=9)
            r += 1
    fig.suptitle('SOCP with explicit margin bounds: maximize sum_i m_i subject to m_min ≤ m_i ≤ m_max\n'
                 'Gray dashed/x = artificial boundary obstacles; purple = real obstacles; black = 10-step query', fontsize=13)
    fig.tight_layout(rect=[0.02, 0.01, 1.0, 0.965])
    fig_path = OUT / 'narrow_gap_m_bounds_K_gamma_6x3.png'
    fig.savefig(fig_path, dpi=175)
    plt.close(fig)

    csv_path = OUT / 'narrow_gap_m_bounds_K_gamma_6x3.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    json_path = OUT / 'narrow_gap_m_bounds_K_gamma_6x3.json'
    json_path.write_text(json.dumps({
        'formulation': 'maximize sum_i m_i with explicit lower and upper face margin bounds m_min <= m_i <= m_max',
        'R': R, 'rho_art': RHO_ART, 'trajectory': TRAJ.tolist(), 'real_obstacles': REAL_OBS,
        'K_values': K_VALUES, 'gammas': GAMMAS,
        'row_modes': [{'label': lab, 'm_min': mn, 'm_max': mx} for lab, mn, mx in ROW_MODES],
        'rows': rows,
        'note': 'When m_max is small, the optimizer clips the margins of both real and artificial faces. This can create a visibly tube-like corridor compared with the unbounded max-margin wedge.'
    }, indent=2))
    readme = OUT / 'README.md'
    readme.write_text('''# Margin-bounded SOCP 6x3 narrow-gap figure

This experiment adds explicit bounds

- `m_i >= m_min`
- `m_i <= m_max`

into the face-wise max-margin SOCP surrogate.  Rows are `K=4,8,16`, each repeated for two margin-bound regimes:

- `loose m-bounds`: `m_min=0.03`, `m_max=5.00`
- `tight m-bounds`: `m_min=0.03`, `m_max=0.40`

Columns are `gamma=0.3,0.5,0.8`.  Artificial obstacles are drawn as gray dashed circles with x markers.

The main expected behavior is that a sufficiently small `m_max` clips the half-width of the verifier faces and can visually turn the sharp wedge into a more tube-like polytope.
''')
    bundle = OUT / 'pillar3_m_bounds_demo_bundle.zip'
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in [fig_path, csv_path, json_path, readme, Path('/mnt/data/pillar3_m_bounds_6x3.py')]:
            z.write(p, p.name)
    print(fig_path)
    print(csv_path)
    print(json_path)
    print(bundle)

if __name__ == '__main__':
    main()
