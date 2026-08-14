"""STAIRCASE-ID QUOTA EXPANSION (user design 2026-07-05) — separate module from grid_expand2.

Base: temp 2.0 · ell 0.5 · enc_mult 0.5 · β 0.1 · s 0.9 · N 64 · lr 2e-4 cosine · α 0 · inner 12×128.
Harvest rules on the positive buffer, keyed by staircase_id (sid):
  0. PRE: roll the pretrained policy (n=25/γ, temp 1.0) → EXISTING = sids of valid2-reached trajectories.
  1. EXISTING sid → its windows may hold ≤5% of the buffer (evict its oldest beyond the cap).
  2. NEW sid → registered on first valid2 harvest; UNCAPPED for 100 iterations, then graduates to EXISTING.
  3. STALENESS: every 10 iters, any non-immune sid present in the buffer for 10 consecutive iters is BANNED
     (windows evicted, future adds rejected) — diagonal ids die out, forcing rollouts toward new data.
  4. WAIT: if the buffer < batch size, the update is SKIPPED (drought) — exploration continues (discovery is
     sampling-driven, not update-driven); droughts are counted and reported.
Outputs: history.json (val2/γ, coverage-new-ids, buffer strata, droughts, bans), ckpt_1000, stats figure.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import grid_scene as GS
import grid_rollout as GR
import grid_metrics as GM
import grid_metrics2 as GM2
import grid_expand as GE
import hp_arch_sweep as ARCH
from uncertainty import GPUncertainty

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GAMMAS = (0.5, 1.0, 0.1)


def ham(a, b):
    """Raw Hamming distance between R/U words; one adjacent transposition = 2. 'Hamming-1 neighborhood' ≤ 2."""
    sa, sb = str(a), str(b)
    return sum(x != y for x, y in zip(sa, sb)) + abs(len(sa) - len(sb))


class QuotaBuffer:
    """Positive windows keyed by staircase-id with the user's quota/immunity/ban rules (PER-γ instance).
    UPGRADE (user 2026-07-05): a 'new' id within Hamming≤2 (one transposition) of an existing/banned id
    INHERITS the 5% cap + 10-iter ban rule (no immunity) — near-diagonal variants don't get the free ride."""
    def __init__(self, existing_ids, cap_frac=0.05, immunity=100):
        self.core = set(existing_ids)          # FIX-D: Hamming inheritance consults ONLY this frozen core
        self.G = self.L = self.H = self.U = None
        self.sid = np.zeros(0, dtype=object)
        self.existing = set(existing_ids)
        self.new_reg = {}                      # sid -> registration iter (immune while it-reg < immunity)
        self.banned = set()
        self.streak = {}                       # sid -> consecutive iters present
        self.cap_frac = cap_frac
        self.immunity = immunity
        self.rejected_banned = 0

    def n(self):
        return 0 if self.U is None else self.U.shape[0]

    def _keep(self, mask):
        if self.U is None:
            return
        idx = torch.as_tensor(np.nonzero(mask)[0])
        self.G, self.L, self.H, self.U = self.G[idx], self.L[idx], self.H[idx], self.U[idx]
        self.sid = self.sid[mask]

    def add(self, sid, G, L, H, U, it):
        if sid in self.banned:
            self.rejected_banned += 1
            return "banned"
        kind = "existing"
        if sid not in self.existing and sid not in self.new_reg:
            near = any(ham(sid, e) <= 2 for e in self.core)
            if near:                                            # Hamming-1 neighbor → inherits cap+ban, no immunity
                self.existing.add(sid)
                kind = "near-existing"
            else:
                self.new_reg[sid] = it                          # genuinely new: uncapped for `immunity` iters
                kind = "NEW"
        elif sid in self.new_reg and it - self.new_reg[sid] < self.immunity:
            kind = "new-immune"
        cat = lambda a, b: b if a is None else torch.cat([a, b], 0)
        self.G, self.L, self.H, self.U = cat(self.G, G), cat(self.L, L), cat(self.H, H), cat(self.U, U)
        self.sid = np.concatenate([self.sid, np.array([sid] * G.shape[0], dtype=object)])
        # enforce ≤5% for non-immune ids
        if kind == "existing" or (sid in self.new_reg and it - self.new_reg[sid] >= self.immunity):
            cap = max(1, int(self.cap_frac * self.n()))
            where = np.nonzero(self.sid == sid)[0]
            if len(where) > cap:
                drop = set(where[:len(where) - cap].tolist())    # evict OLDEST of this sid
                self._keep(np.array([i not in drop for i in range(self.n())]))
        return kind

    def tick(self, it):
        """Streak accounting + bans (every 10 iters). Immune ids can't be banned."""
        present = set(self.sid.tolist())
        for s in present:
            self.streak[s] = self.streak.get(s, 0) + 1
        for s in list(self.streak):
            if s not in present:
                self.streak[s] = 0
        newly_banned = []
        if it % 10 == 0:
            for s, k in self.streak.items():
                immune = s in self.new_reg and it - self.new_reg[s] < self.immunity
                if k >= 10 and not immune and s not in self.banned:
                    self.banned.add(s)
                    newly_banned.append(s)
            for s_ in newly_banned:                      # FIX-D: retention floor — trim to 5%, do NOT evict
                cap = max(1, int(self.cap_frac * self.n()))
                where = np.nonzero(self.sid == s_)[0]
                if len(where) > cap:
                    drop = set(where[:len(where) - cap].tolist())
                    self._keep(np.array([i not in drop for i in range(self.n())]))
        return newly_banned

    def strata(self):
        vals, counts = np.unique(self.sid, return_counts=True) if self.n() else ([], [])
        n_new = sum(c for v, c in zip(vals, counts) if v in self.new_reg and v not in self.existing)
        return dict(n=self.n(), ids=len(vals), new_ids=len([v for v in vals if v in self.new_reg]),
                    n_new_windows=int(n_new), banned=len(self.banned))


def measure(policy, env, n=25):
    out = {}
    gl = env.goal.detach().cpu().numpy()
    for g in GAMMAS:
        torch.manual_seed(1)
        paths = GR.deploy_many(policy, env, g, n, T=250, temp=1.0, nfe=8, device=DEV)
        ok = reach = 0
        sids = set()
        for p in paths:
            P = np.asarray(p, np.float32)
            reach += int(np.linalg.norm(P[-1, :2] - gl) < 0.5)
            v = GM2.traj_valid2(P, env, g)
            v = v[0] if isinstance(v, tuple) else bool(v)
            ok += int(v)
            if v:
                s = GM.staircase_id(P)
                if s is not None:
                    sids.add(s)
        out[g] = dict(val2=ok / n, reach=reach / n, sids=sids)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--outdir", default=os.path.join(HERE, "results", "hp_quota"))
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--ell", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--s", type=float, default=0.9)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--enc-mult", type=float, default=0.5)
    ap.add_argument("--inner", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    env = GS.make_grid()
    pol, _ = ARCH.load_arch(os.path.join(HERE, "results", "hp_arch", "res2w256_ft.pt"), device=DEV)

    # --- step 0: existing repertoire of the pretrained policy ---
    m0 = measure(pol, env)
    print("[pre] val2 " + " ".join(f"γ{g}:{m0[g]['val2']*100:.0f}%" for g in GAMMAS) +
          f" · EXISTING ids/γ: {[len(m0[g]['sids']) for g in GAMMAS]}", flush=True)

    bufs = {g: QuotaBuffer(set(m0[g]["sids"])) for g in GAMMAS}   # PER-γ quota books (user upgrade #2)
    field = list(pol.trunk.parameters()) + list(pol.head.parameters())
    enc = pol.encoder_modules()
    opt = torch.optim.Adam([{"params": field, "lr": a.lr}, {"params": enc, "lr": a.lr * a.enc_mult}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters)
    unc = GPUncertainty(kernel="rbf", lengthscale=a.ell, lam=1e-2, normalize=True)
    qbuf = None
    hist = []
    droughts = upd = 0
    new_timeline = []
    for it in range(1, a.iters + 1):
        g = GAMMAS[(it - 1) % 3]
        feats = GE._buffer_feat(pol, qbuf, "phi_s", a.s, 384, DEV)
        if feats is not None:
            unc.set_buffer(feats)
        out = GR.fm_deploy(pol, env, g, T=250,
                           tilt=dict(unc=unc, beta=a.beta, N=a.N, s=a.s, broad=0, feature="phi_s",
                                     temp=a.temp, churn=0.05, safe_filter=True),
                           nfe=6, record=True, verify_fn=GM2.window_label_cheap, device=DEV)
        if out["recs"]:
            G, L, H, U = GE._to_t(out["recs"])
            qbuf = GE._cat(qbuf, G[::3], L[::3], H[::3], U[::3], cap=4096)
            if out["reached"]:
                v = GM2.traj_valid2(np.asarray(out["path"], np.float32), env, g)
                if (v[0] if isinstance(v, tuple) else bool(v)):
                    sid = GM.staircase_id(np.asarray(out["path"], np.float32))
                    if sid is not None:
                        kind = bufs[g].add(sid, G, L, H, U, it)
                        if kind == "NEW":
                            new_timeline.append((it, float(g), str(sid)))
                            print(f"  it{it}: NEW id (γ{g}) registered ({len(new_timeline)} total)", flush=True)
        for gg in GAMMAS:
            bufs[gg].tick(it)
        live = [b for b in bufs.values() if b.n() > 0]
        total_n = sum(b.n() for b in live)
        if total_n >= a.batch:
            pol.train()
            for _ in range(a.inner):
                parts = []
                for b in live:
                    nb = max(1, round(a.batch * b.n() / total_n))
                    ii = torch.randint(0, b.n(), (nb,))
                    parts.append((b.G[ii], b.L[ii], b.H[ii], b.U[ii]))
                Gb, Lb, Hb, Ub = (torch.cat([p[i] for p in parts], 0).to(DEV) for i in range(4))
                loss = pol.cfm_loss(Ub, pol.ctx_from(Gb, Lb, Hb))
                opt.zero_grad(); loss.backward(); opt.step()
            pol.eval(); upd += 1
        else:
            droughts += 1                                        # WAIT mode: explore, don't update
        sched.step()
        if it % 100 == 0:
            mm = measure(pol, env)
            agg = [b.strata() for b in bufs.values()]
            st = dict(n=sum(x["n"] for x in agg), ids=sum(x["ids"] for x in agg),
                      new_ids=sum(x["new_ids"] for x in agg), n_new_windows=sum(x["n_new_windows"] for x in agg),
                      banned=sum(x["banned"] for x in agg))
            rec = dict(it=it, val2={str(g): mm[g]["val2"] for g in GAMMAS},
                       reach={str(g): mm[g]["reach"] for g in GAMMAS},
                       new_registered=len(new_timeline), droughts=droughts, updates=upd, **st)
            hist.append(rec)
            print(f"it{it:04d}: val2 " + "/".join(f"{mm[g]['val2']*100:.0f}" for g in GAMMAS) +
                  f" reach " + "/".join(f"{mm[g]['reach']*100:.0f}" for g in GAMMAS) +
                  f" | buf {st['n']} win · {st['ids']} ids ({st['new_ids']} new, {st['banned']} banned) | "
                  f"NEW-ids {len(new_timeline)} · droughts {droughts} · upd {upd}", flush=True)
            json.dump(dict(hist=hist, new_timeline=new_timeline, pre_val2={str(g): m0[g]["val2"] for g in GAMMAS},
                           banned={str(g): [str(x) for x in bufs[g].banned] for g in GAMMAS}),
                      open(os.path.join(a.outdir, "history.json"), "w"))
    torch.save({"state_dict": pol.state_dict(), "variant": "res2w256", "quota_iters": a.iters},
               os.path.join(a.outdir, f"ckpt_{a.iters}.pt"))
    print(f"saved {a.outdir}/ckpt_{a.iters}.pt · NEW ids {len(new_timeline)} · "
          f"banned {sum(len(b.banned) for b in bufs.values())} · droughts {droughts}/{a.iters}", flush=True)


if __name__ == "__main__":
    main()
