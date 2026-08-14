"""Bounded, non-replay approval trace for the HP100 ball-protocol port.

This is deliberately not an expansion runner.  It loads the promoted HP100
checkpoint, freezes everything except the final flow head, and executes one
fixed-scene, round-1 diagnostic batch with 16 independent flow-noise lineages.
At every active context it generates K=16 plans, selects B=4 with an empty RBF
posterior, applies the exact clipped H10 SFM verifier, and executes

    argmin_j J_native(U_j) - lambda * m_step(U_j)

over exact-positive B queries only.  If no such query exists, that lineage
terminates NVP without a fallback.  A user-requested diagnostic step bound is
recorded as ``cutoff`` and every output is marked ``enters_replay=false``.

The renderer is trace-only: all candidate segments, labels, and verifier faces
are serialized here before visualization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
from sfm_hp100_ball_core.expansion import RBFPosterior, _counter_seed
import sfm_hp100_parallel_gather_viz as VIZ
import sfm_metrics2 as VERIFY
import sfm_scene as SS


VERSION = "sfm_hp100_ball_approval_smoke_v1"
STATUS = "SFM_HP100_BALL_APPROVAL_SMOKE_COMPLETE"
ROUND = 1
RETRY_BATCH = 0
K = 16
B = 4
H = 10
PARALLEL_EPISODES = 16


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        # Keep exact verifier diagnostics losslessly JSON-native.  These values
        # are never selector inputs; flat lambda rows remain strictly finite.
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    return value


def select_exact_positive(
    results: Sequence, *, step_margin_weight: float,
) -> tuple[int | None, list[float | None]]:
    """Return the local B index minimizing J-lambda*m, or NVP as ``None``."""
    weight = float(step_margin_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("step-margin weight must be finite and nonnegative")
    eligible = []
    scores: list[float | None] = []
    for local, result in enumerate(results):
        if result.error:
            raise RuntimeError("approval smoke refuses a verifier error")
        if result.step_margin is None:
            raise ValueError("every resolved query must provide one-step margin")
        score = float(result.execution_cost) - weight * float(result.step_margin)
        if not math.isfinite(score):
            raise ValueError("execution score must be finite")
        exact_positive = bool(result.valid and result.progress_eligible)
        scores.append(score if exact_positive else None)
        if exact_positive:
            eligible.append((score, local))
    if not eligible:
        return None, scores
    return min(eligible, key=lambda row: (row[0], row[1]))[1], scores


def _load_checkpoint(path: str, expected_sha256: str, device: str):
    actual = sha256_file(path)
    if actual != str(expected_sha256).lower():
        raise RuntimeError(f"checkpoint SHA256 mismatch: {actual} != {expected_sha256}")
    policy, payload = GPS.load_sfm_hp100_policy(path, device=device)
    GPS.configure_head_only_expansion(policy)
    adapter = PORT.HP100ExpansionPolicy(policy)
    trainable = PORT.assert_head_only(adapter)
    policy.eval()
    return adapter, payload, actual, trainable


def _event_status(task_status: str | None) -> str:
    if task_status is None:
        return "active"
    mapping = {"SUCCESS": "success", "COLLISION": "collision"}
    if task_status not in mapping:
        raise RuntimeError(f"unexpected approval-smoke terminal status {task_status!r}")
    return mapping[task_status]


@torch.no_grad()
def collect_trace(
    adapter: PORT.HP100ExpansionPolicy,
    task: PORT.SFMHP100ExpansionTask,
    *,
    gamma: float,
    scenario_id: int,
    diagnostic_steps: int,
    beta: float,
    rbf_lengthscale: float,
    rbf_noise: float,
    step_margin_weight: float,
    seed: int,
) -> tuple[dict, list[dict], list[dict]]:
    """Collect exactly one fixed-scene 16-lineage diagnostic retry batch."""
    if task.fixed_scenario_id != int(scenario_id) or not task.trace_sidecars:
        raise ValueError("approval task must use the declared fixed scene and trace sidecars")
    if int(diagnostic_steps) < 1:
        raise ValueError("diagnostic_steps must be positive")
    if not math.isfinite(float(beta)) or float(beta) <= 0.0:
        raise ValueError("beta must be finite and positive")
    device = next(adapter.parameters()).device
    posterior = RBFPosterior(float(rbf_lengthscale), float(rbf_noise))
    posterior.set_buffer(None)
    states = [
        task.reset(
            float(gamma), replica,
            _counter_seed(seed, "reset", ROUND, RETRY_BATCH, replica),
        )
        for replica in range(PARALLEL_EPISODES)
    ]
    terminal: list[str | None] = [None] * PARALLEL_EPISODES
    events: list[dict] = []
    lambda_rows: list[dict] = []

    for step in range(int(diagnostic_steps)):
        for replica in range(PARALLEL_EPISODES):
            if terminal[replica] is not None:
                continue
            state = states[replica]
            context = task.context(state, float(gamma)).detach().to(device)
            generator = torch.Generator(device=device).manual_seed(
                _counter_seed(seed, "gather", ROUND, RETRY_BATCH, replica, step)
            )
            # Match production exactly: paired_noised_representation=False uses
            # endpoint phi and does not expose or reuse the flow base noise.
            candidates = adapter.sample(
                context, K, generator, base_std=1.0,
            ).detach()
            features = adapter.embed(context, candidates)
            sigma = posterior.sigma(features).detach().cpu()
            selected, selected_sigma, conditional_ess = posterior.acquire(
                features, B, float(beta), generator,
            )
            queried = candidates[selected]
            results = list(task.verify(context, queried, float(gamma)))
            sidecars = task.pop_trace_sidecars()
            if len(results) != B or len(sidecars) != B:
                raise RuntimeError("serial verifier did not return exactly B=4 results")
            chosen_local, scores = select_exact_positive(
                results, step_margin_weight=step_margin_weight,
            )
            chosen_global = None if chosen_local is None else int(selected[chosen_local])

            robot, ped_xy, _ = task.decode_context(context)
            all_K = [
                dict(
                    candidate_id=index,
                    segment=PORT.clipped_plan_states(
                        robot, candidate.detach().cpu().numpy(),
                    )[:, :2],
                    acquisition_sigma=float(sigma[index]),
                )
                for index, candidate in enumerate(candidates)
            ]
            query_rows = []
            context_id = f"r{ROUND}:g{float(gamma):.9g}:b{RETRY_BATCH}:rep{replica}:t{step}"
            for local, (global_id, result, sidecar) in enumerate(zip(
                selected, results, sidecars,
            )):
                exact_positive = bool(result.valid and not result.error)
                execution_eligible = bool(exact_positive and result.progress_eligible)
                row = dict(sidecar)
                row["candidate_id"] = int(global_id)
                row["execution_eligible"] = execution_eligible
                row["selection_score"] = scores[local]
                query_rows.append(row)
                lambda_rows.append(dict(
                    gamma=float(gamma), context_id=context_id,
                    candidate_id=int(global_id), exact_positive=exact_positive,
                    execution_eligible=execution_eligible,
                    native_cost=float(result.execution_cost),
                    step_margin=float(result.step_margin),
                    round=ROUND, retry_batch=RETRY_BATCH, replica=replica,
                    lineage_id=f"noise_{replica:02d}", step=step,
                    selected_sigma=float(selected_sigma[local]),
                    chosen=bool(chosen_local == local),
                ))

            if chosen_local is None:
                status = "nvp"
                terminal[replica] = status
            else:
                states[replica] = task.advance(state, queried[chosen_local])
                status = _event_status(task.terminal(states[replica]))
                if status != "active":
                    terminal[replica] = status
                elif step + 1 == int(diagnostic_steps):
                    status = "cutoff"
                    terminal[replica] = status

            events.append(dict(
                round=ROUND, gamma=float(gamma), retry_batch=RETRY_BATCH,
                replica=replica, lineage_id=f"noise_{replica:02d}", step=step,
                scenario_id=int(scenario_id), state=np.asarray(robot, np.float32),
                ped_xy=np.asarray(ped_xy, np.float32), all_K=all_K,
                selected_ids=list(map(int, selected)), query_rows=query_rows,
                executed_id=chosen_global, episode_status=status,
                committed_success=False,
                acquisition=dict(
                    reference_rows=0, round1_empty_rbf_bootstrap=True,
                    beta=float(beta), marginal_sigma=list(map(float, sigma)),
                    selected_sigma=list(map(float, selected_sigma)),
                    conditional_normalized_ess=list(map(float, conditional_ess)),
                ),
                selector=dict(
                    formula="J_native - lambda * m_step",
                    lambda_value=float(step_margin_weight),
                    chosen_local=chosen_local,
                ),
            ))

    if any(value is None for value in terminal):
        raise RuntimeError("bounded approval trace left an unterminated lineage")
    metadata = dict(
        schema_version=VIZ.SCHEMA_VERSION, producer_version=VERSION,
        diagnostic_only=True, enters_replay=False,
        cutoff_semantics=(
            "diagnostic wall only; cutoff is neither timeout nor terminal success "
            "and no row is eligible for replay"
        ),
        K=K, B=B, H=H, parallel_episodes=PARALLEL_EPISODES,
        expected_artificial_faces=VERIFY.ARTIFICIAL_FACES,
        verifier_contract=VERIFY.verifier_manifest(),
        scene_profile=SS.scene_profile(task.scene_profile),
        fixed_scenario_id=int(scenario_id), gamma=float(gamma),
        goal=SS.GOAL.astype(float).tolist(),
        plot_bounds=[SS.TASK_LO, SS.TASK_HI, SS.TASK_LO, SS.TASK_HI],
        pedestrian_radius=float(SS.R_PED), diagnostic_steps=int(diagnostic_steps),
        acquisition=dict(
            posterior="empty RBF round-1 bootstrap", reference_rows=0,
            lengthscale=float(rbf_lengthscale), noise=float(rbf_noise),
            beta=float(beta), paired_noised_representation=False, note=(
                "first marginal sigma is constant, hence first-draw ESS/K=1; "
                "later B draws only reflect within-K conditional diversity"
            ),
        ),
        selector=dict(
            formula="J_native - lambda * m_step over exact positives only",
            lambda_value=float(step_margin_weight), no_fallback=True,
        ),
    )
    return _jsonable(metadata), _jsonable(events), _jsonable(lambda_rows)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--expected-checkpoint-sha256", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--device", default="mps")
    value.add_argument("--scene-profile", default=PORT.DEFAULT_SCENE_PROFILE,
                       choices=SS.SCIENTIFIC_EVAL_PROFILES)
    value.add_argument("--scenario-id", type=int, required=True)
    value.add_argument("--gamma", type=float, default=0.5, choices=SS.GAMMAS)
    value.add_argument("--diagnostic-steps", type=int, default=8)
    value.add_argument("--beta", type=float, default=5.0e-4)
    value.add_argument("--rbf-lengthscale", type=float, required=True)
    value.add_argument("--rbf-noise", type=float, default=1.0e-2)
    value.add_argument("--execution-step-margin-weight", type=float, default=70_000.0)
    value.add_argument("--seed", type=int, default=2)
    value.add_argument("--fps", type=int, default=6)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output root: {output}")
    output.mkdir(parents=True)
    adapter, payload, checkpoint_sha, trainable = _load_checkpoint(
        args.checkpoint, args.expected_checkpoint_sha256, args.device,
    )
    task = PORT.SFMHP100ExpansionTask(
        scene_profile=args.scene_profile, fixed_scenario_id=args.scenario_id,
        trace_sidecars=True,
    ).attach_context_encoder(adapter.policy)
    metadata, events, lambda_rows = collect_trace(
        adapter, task, gamma=args.gamma, scenario_id=args.scenario_id,
        diagnostic_steps=args.diagnostic_steps, beta=args.beta,
        rbf_lengthscale=args.rbf_lengthscale, rbf_noise=args.rbf_noise,
        step_margin_weight=args.execution_step_margin_weight, seed=args.seed,
    )
    metadata["checkpoint"] = dict(
        path=str(Path(args.checkpoint).resolve()), sha256=checkpoint_sha,
        scientific_status=payload.get("scientific_status"),
        trainable_parameter_names=trainable,
        trainable_parameter_count=sum(
            parameter.numel() for parameter in adapter.parameters()
            if parameter.requires_grad
        ),
    )
    trace = output / "approval_trace.json"
    candidates = output / "lambda_candidates.jsonl"
    _write_json(trace, dict(metadata=metadata, events=events))
    _write_jsonl(candidates, lambda_rows)
    render = VIZ.render_bundle(
        trace, output / "render", round_index=ROUND, gamma=args.gamma,
        retry_batch=RETRY_BATCH, fps=args.fps,
    )
    mixed = [
        event for event in events
        if 0 < sum(int(row["result"]["y"]) for row in event["query_rows"]) < B
    ]
    if mixed:
        diagnostic = min(mixed, key=VIZ.event_identity)
        mixed_png, mixed_pdf = VIZ.render_candidate_specific(
            metadata, diagnostic,
            output / "render" / "candidate_specific_B_mixed.png",
            output / "render" / "candidate_specific_B_mixed.pdf",
        )
        render["mixed_candidate_png"] = mixed_png
        render["mixed_candidate_pdf"] = mixed_pdf
    marker = output / "APPROVAL_SMOKE_COMPLETE.json"
    _write_json(marker, dict(
        status=STATUS, producer_version=VERSION, diagnostic_only=True,
        enters_replay=False, checkpoint_sha256=checkpoint_sha,
        trace=dict(path=str(trace), sha256=sha256_file(trace)),
        lambda_candidates=dict(
            path=str(candidates), sha256=sha256_file(candidates), rows=len(lambda_rows),
        ),
        render={key: str(value) for key, value in render.items()},
    ))
    print(json.dumps(dict(status=STATUS, output=str(output), marker=str(marker))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
