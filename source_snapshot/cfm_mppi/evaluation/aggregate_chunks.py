"""Aggregate episodes.jsonl from several eval chunk dirs into one summary with
per-(method,gamma) success/collision/clearance + paired McNemar vs Mizuta."""
from __future__ import annotations
import argparse, json, math
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--out", default=None)
    cli = p.parse_args()
    rows = []
    for d in cli.dirs:
        try:
            for l in open(f"{d}/episodes.jsonl"):
                rows.append(json.loads(l))
        except FileNotFoundError:
            pass

    def grp(m, g=None):
        return {r["episode"]: r for r in rows
                if r["method"] == m and (g is None or abs((r.get("gamma") or -99) - g) < 1e-6)}

    miz = grp("mizuta_cfm_mppi")
    variants = []
    seen = set()
    for r in rows:
        key = (r["method"], r.get("gamma"))
        if key not in seen:
            seen.add(key); variants.append(key)

    lines = []
    lines.append(f"{'method':<24}{'gamma':>6}{'n':>5}{'succ%':>7}{'coll%':>7}{'clrMed':>8}   vs Mizuta McNemar(o/m) p")
    for (m, g) in variants:
        d = grp(m, g); eps = sorted(d)
        if not eps:
            continue
        su = np.array([d[e]["success"] for e in eps]); co = np.array([d[e]["collision"] for e in eps])
        cl = np.array([d[e]["min_clearance"] for e in eps]); cl = cl[np.isfinite(cl)]
        clm = float(np.median(cl)) if cl.size else float("nan")
        ms = np.array([miz[e]["success"] for e in eps if e in miz])
        os_ = np.array([d[e]["success"] for e in eps if e in miz])
        if ms.size:
            n01 = int(np.sum(os_ & ~ms)); n10 = int(np.sum(~os_ & ms)); n = n01 + n10
            pv = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) * 0.5**n for i in range(min(n01, n10) + 1)))
            mc = f"{n01}/{n10} p={pv:.2g}"
        else:
            mc = "--"
        gs = "-" if g is None else f"{g:.2g}"
        lines.append(f"{m:<24}{gs:>6}{len(eps):>5}{100*su.mean():>7.1f}{100*co.mean():>7.1f}{clm:>8.3f}   {mc}")
    text = "\n".join(lines)
    print(text)
    if cli.out:
        with open(cli.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
