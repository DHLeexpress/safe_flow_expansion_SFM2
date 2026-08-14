"""Compare final planned-window branch forests for two SFM checkpoints."""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import sfm_b1_d_branch_viz as DB
import sfm_b1_full_episode_viz as FV


STATUS = "SFM_B1_BRANCH_COMPARISON_COMPLETE"


def summarize(bundle):
    traces = list(bundle["traces"])
    outcomes = list(bundle["outcomes"])
    positive = sum(
        row["executed_label"] == "verifier_positive" for row in traces
    )
    negative = sum(
        row["executed_label"] == "verifier_negative" for row in traces
    )
    resolved = positive + negative
    return {
        "contexts": len(traces),
        "executed_positive": positive,
        "executed_negative": negative,
        "executed_positive_fraction": (
            float(positive / resolved) if resolved else None
        ),
        "success": sum(bool(row["success"]) for row in outcomes),
        "collision": sum(bool(row["collision"]) for row in outcomes),
        "timeout": sum(bool(row["timeout"]) for row in outcomes),
        "episodes": len(outcomes),
    }


def _load(path):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("status") != "SFM_B1_FULL_EPISODE_LABEL_AUDIT_COMPLETE":
        raise ValueError(f"not a completed branch audit: {path}")
    return bundle


def render(
    pretrained_trace, expanded_trace, output_png, output_json,
    *, expanded_label="partial expanded",
):
    pretrained = _load(pretrained_trace)
    expanded = _load(expanded_trace)
    for key in ("scenarios", "gammas", "environment", "sample_seed", "audit_seed"):
        if pretrained[key] != expanded[key]:
            raise ValueError(f"comparison contract differs at {key}")
    scenarios = tuple(map(int, pretrained["scenarios"]))
    gammas = tuple(map(float, pretrained["gammas"]))
    if len(scenarios) != 3 or len(gammas) != 7:
        raise ValueError("comparison requires three scenarios and seven gammas")

    models = (("pretrained", pretrained), (expanded_label, expanded))
    figure, axes = plt.subplots(6, 7, figsize=(23.4, 18.0))
    figure.subplots_adjust(
        left=.055, right=.82, bottom=.025, top=.96, wspace=.025, hspace=.04,
    )
    for column, gamma in enumerate(gammas):
        figure.text(
            .055 + (.765 / 7) * (column + .5), .975,
            f"$\\gamma={gamma:g}$", ha="center", va="center", fontsize=10,
        )

    reports = {}
    for model_index, (label, bundle) in enumerate(models):
        index = FV._index(bundle["traces"])
        reports[label] = summarize(bundle)
        for scenario_index, scenario in enumerate(scenarios):
            row = model_index * len(scenarios) + scenario_index
            figure.text(
                .018, .96 - (.935 / 6) * (row + .5),
                f"{label}\nepisode {scenario}",
                ha="center", va="center", rotation=90, fontsize=8,
            )
            for column, gamma in enumerate(gammas):
                rows = index[(scenario, round(gamma, 8))]
                DB.draw_cell(axes[row, column], rows, max(rows))

    figure.legend(
        handles=DB._legend(), loc="center left", bbox_to_anchor=(.835, .66),
        frameon=False, fontsize=8,
    )
    text = []
    for label, report in reports.items():
        text.extend([
            label,
            f"verified D: {report['executed_positive']}/"
            f"{report['contexts']} "
            f"({report['executed_positive_fraction']:.1%})",
            f"outcomes S/C/T: {report['success']}/"
            f"{report['collision']}/{report['timeout']}",
            "",
        ])
    text.extend([
        "Fixed scenarios, gamma values, and",
        "proposal x0 streams are shared.",
        "Contexts diverge after the first",
        "closed-loop action.",
    ])
    figure.text(
        .835, .42, "\n".join(text), ha="left", va="top", fontsize=8,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    figure.savefig(output_png, dpi=165, bbox_inches="tight")
    plt.close(figure)
    report = {
        "status": STATUS,
        "pretrained_trace": os.path.abspath(pretrained_trace),
        "expanded_trace": os.path.abspath(expanded_trace),
        "expanded_label": expanded_label,
        "comparison_contract": {
            "scenarios": list(scenarios),
            "gammas": list(gammas),
            "sample_seed": pretrained["sample_seed"],
            "audit_seed": pretrained["audit_seed"],
            "closed_loop_caveat": (
                "proposal x0 streams match by cell and step, but checkpoint-"
                "dependent first actions make later contexts different"
            ),
        },
        "models": reports,
        "png": os.path.abspath(output_png),
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    temporary = os.path.abspath(output_json) + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    os.replace(temporary, os.path.abspath(output_json))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-trace", required=True)
    parser.add_argument("--expanded-trace", required=True)
    parser.add_argument("--expanded-label", default="partial expanded")
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    render(
        args.pretrained_trace,
        args.expanded_trace,
        args.output_png,
        args.output_json,
        expanded_label=args.expanded_label,
    )


if __name__ == "__main__":
    main()
