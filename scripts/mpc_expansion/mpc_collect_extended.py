"""Extended archive collection: v2 MPC-cost selector + raw-observation capture.

Runs the unmodified expansion archive runner under two scoped monkeypatches:
``install_v2_selector`` (acquisition executes the tuned MPC-cost rule;
lam/rho/r_eff/sigma from MPC_* env, defaults 4.0/1.1/0.45/0.1) and
``install_raw_obs_capture`` (every replan's raw encoder inputs — the Hp100
raster stack, low5, and GRU control history — stream into
``<output>/raw_obs/`` shards with inline bitwise re-encode audits), so a
later declared phase can flow gradients into ``grid_projection`` and other
condition-encoder surfaces.  Unlike the earlier collect driver this one also
persists ``selector_shadow.jsonl`` so mode tagging does not need a trace
join.  Spawn-guarded: verifier workers re-import this file as the main
module path.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path.home() / "projects/safe_flow_expansion_SFM2-claude-cfc09ad"
sys.path.insert(0, str(ROOT / "source_snapshot/overnight_run_07_12_sfm"))
import sfm_hp100_predictive_execution_v2 as V2  # noqa: E402
import sfm_hp100_expansion_archive as ARCH  # noqa: E402
import sfm_hp100_raw_obs_capture as RAW  # noqa: E402

PARAMS = V2.MPCRuleParams(
    lam=float(os.environ.get("MPC_LAM", "4.0")),
    rho=float(os.environ.get("MPC_RHO", "1.1")),
    r_eff=float(os.environ.get("MPC_R_EFF", "0.45")),
    sigma_len=float(os.environ.get("MPC_SIGMA", "0.1")),
)
FLUSH_EVERY = int(os.environ.get("RAW_OBS_FLUSH_EVERY", RAW.DEFAULT_FLUSH_EVERY))
VERIFY_EVERY = int(os.environ.get("RAW_OBS_VERIFY_EVERY", RAW.DEFAULT_VERIFY_EVERY))


def _run() -> int:
    argv = sys.argv[1:]
    output = Path(argv[argv.index("--output") + 1])
    # The shard writer must not create <output> before ARCH.run does — the
    # archive runner refuses a pre-existing output directory. Stream shards
    # into a sibling temp dir and move them in after the run.
    raw_tmp = output.parent / f".{output.name}_raw_obs_tmp"
    shadow: list = []
    with V2.install_v2_selector(PARAMS, shadow):
        with RAW.install_raw_obs_capture(
            raw_tmp,
            flush_every=FLUSH_EVERY, verify_every=VERIFY_EVERY,
        ) as recorder:
            code = ARCH.main(argv)
    output.mkdir(parents=True, exist_ok=True)
    final_raw = output / "raw_obs"
    if raw_tmp.exists() and not final_raw.exists():
        raw_tmp.rename(final_raw)
    (output / "EXECUTION_RULE.json").write_text(json.dumps({
        "execution_rule": "predictive_mpc_v2",
        "authoritative": False,
        "params": {
            "lam": PARAMS.lam,
            "rho": PARAMS.rho,
            "r_eff": PARAMS.r_eff,
            "sigma_len": PARAMS.sigma_len,
        },
        "raw_obs_capture": {
            "version": RAW.VERSION,
            "records": recorder.records,
            "inline_bitwise_verifications": recorder.verified,
            "flush_every": FLUSH_EVERY,
            "verify_every": VERIFY_EVERY,
        },
        "selector_shadow_calls": len(shadow),
    }, indent=2) + "\n")
    (output / "selector_shadow.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in shadow)
    )
    return code


# Spawn verifier workers re-import this file as the main-module path; every
# effectful statement must stay behind the guard or each worker re-runs the
# collection and the pool collapses.
if __name__ == "__main__":
    sys.exit(_run())
