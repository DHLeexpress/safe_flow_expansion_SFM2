"""Run one expansion recipe with the v2 MPC-cost acquisition selector.

Scoped monkeypatch around the unmodified round orchestrator: gather uses
select_predictive_mpc, training/checkpointing are byte-identical to the
declared recipe path.  Rule parameters come from the MPC_LAM / MPC_RHO /
MPC_R_EFF / MPC_SIGMA environment variables (defaults are the 2-pass tuned
values lam=4, rho=1.1, r_eff=0.45, sigma_len=0.1); the actual values are
recorded in EXECUTION_RULE.json so these arms can never be mistaken for
authoritative-rule arms.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path.home() / "projects/safe_flow_expansion_SFM2-claude-cfc09ad"
sys.path.insert(0, str(ROOT / "source_snapshot/overnight_run_07_12_sfm"))
import sfm_hp100_predictive_execution_v2 as V2  # noqa: E402
import sfm_hp100_expansion_round as ROUND  # noqa: E402

PARAMS = V2.MPCRuleParams(
    lam=float(os.environ.get("MPC_LAM", "4.0")),
    rho=float(os.environ.get("MPC_RHO", "1.1")),
    r_eff=float(os.environ.get("MPC_R_EFF", "0.45")),
    sigma_len=float(os.environ.get("MPC_SIGMA", "0.1")),
)


def _run() -> int:
    argv = sys.argv[1:]
    output = Path(argv[argv.index("--output") + 1])
    shadow: list = []
    with V2.install_v2_selector(PARAMS, shadow):
        code = ROUND.main(argv)
    output.mkdir(parents=True, exist_ok=True)
    (output / "EXECUTION_RULE.json").write_text(json.dumps({
        "execution_rule": "predictive_mpc_v2",
        "authoritative": False,
        "params": {
            "lam": PARAMS.lam,
            "rho": PARAMS.rho,
            "r_eff": PARAMS.r_eff,
            "sigma_len": PARAMS.sigma_len,
        },
        "tuning": "mpc_rule/grid_pass1 + grid_pass2 (56-lineage validation: 52 success / 4 NVP / 0 collision)",
        "selector_shadow_calls": len(shadow),
    }, indent=2) + "\n")
    (output / "selector_shadow.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in shadow)
    )
    return code


# The verifier pool uses the spawn start method: workers re-import this file
# as the main-module path, so everything effectful must sit behind the guard
# or every worker re-runs the recipe and the pool collapses.
if __name__ == "__main__":
    sys.exit(_run())
