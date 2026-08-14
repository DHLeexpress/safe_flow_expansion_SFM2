from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

import sfm_hp100_ball_approval_smoke as S
from sfm_hp100_ball_core.expansion import Verification
import sfm_hp100_dynamics as DYN
import sfm_hp100_parallel_gather_viz as VIZ


def _faces():
    return [
        dict(
            a=[float(np.cos(2 * np.pi * index / 16)),
               float(np.sin(2 * np.pi * index / 16))],
            m=2.0, kind="artificial", label=f"outer_{index}",
            coefficient=1.0, feasible=True, interval=None,
        )
        for index in range(16)
    ]


@dataclass
class _State:
    robot: np.ndarray
    status: str | None = None


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 2)

    def sample_with_base(self, context, count, generator, base_std=1.0):
        del context, generator, base_std
        candidates = torch.zeros(int(count), 10, 2)
        candidates[:, :, 0] = torch.linspace(-0.4, 0.4, int(count))[:, None]
        return candidates, torch.zeros_like(candidates)

    def sample(self, context, count, generator, base_std=1.0):
        return self.sample_with_base(context, count, generator, base_std)[0]

    def embed(self, context, candidates, base=None):
        del context, base
        first = candidates[:, 0, 0]
        return torch.stack((first, torch.ones_like(first)), dim=1)


class _Task:
    fixed_scenario_id = 250_007
    trace_sidecars = True
    scene_profile = "double_density_velocity_ood"

    def __init__(self, *, positive=True):
        self.positive = bool(positive)
        self._sidecars = []

    def reset(self, gamma, episode, seed):
        del gamma, episode, seed
        return _State(np.zeros(4, np.float32))

    def context(self, state, gamma):
        del gamma
        return torch.from_numpy(state.robot.copy())

    def decode_context(self, context):
        return (
            context.detach().cpu().numpy().astype(np.float32),
            np.asarray([[3.0, 3.0]], np.float32),
            np.zeros((1, 2), np.float32),
        )

    def verify(self, context, candidates, gamma):
        del gamma
        robot, _, _ = self.decode_context(context)
        results, sidecars = [], []
        for local, candidate in enumerate(candidates):
            segment = S.PORT.clipped_plan_states(robot, candidate)[:, :2]
            cost = float(10 + local)
            margin = float(local) * 0.01
            results.append(Verification(
                valid=self.positive, hp_eligible=True, margin=1.0,
                execution_cost=cost, progress_eligible=True, error=False,
                step_margin=margin,
            ))
            sidecars.append(dict(
                candidate_id=local, execution_eligible=self.positive,
                result=dict(
                    resolved=True, y=int(self.positive), full_h=True,
                    terminal_step=10, taskspace=True, collision_free=True,
                    certificate=self.positive, segment=segment,
                    faces=(_faces() if self.positive else []), diagnostics={},
                ),
            ))
        self._sidecars.append(sidecars)
        return results

    def pop_trace_sidecars(self):
        return self._sidecars.pop(0)

    def advance(self, state, candidate):
        state.robot = DYN.step_numpy(state.robot, candidate[0].detach().cpu().numpy())
        return state

    def terminal(self, state):
        return state.status


def test_selector_is_exact_positive_only_and_uses_native_cost_minus_margin():
    rows = [
        Verification(True, True, 1., 10., step_margin=0.),
        Verification(True, True, 1., 12., step_margin=.1),
        Verification(False, True, 1., -100., step_margin=100.),
    ]
    chosen, scores = S.select_exact_positive(rows, step_margin_weight=100.)
    assert chosen == 1
    assert scores == [10., 2., None]

    chosen, _ = S.select_exact_positive(
        [Verification(False, True, 1., 1., step_margin=0.)],
        step_margin_weight=1.,
    )
    assert chosen is None


def test_nonfinite_verifier_diagnostics_are_preserved_as_explicit_strings():
    value = S._jsonable(dict(
        scalar=np.float32(np.inf), nested=torch.tensor([float("-inf"), float("nan")]),
    ))
    assert value == {"scalar": "Infinity", "nested": ["-Infinity", "NaN"]}


def test_bounded_trace_marks_cutoff_nonreplay_and_empty_rbf_bootstrap():
    metadata, events, lambda_rows = S.collect_trace(
        _Policy(), _Task(positive=True), gamma=.5, scenario_id=250_007,
        diagnostic_steps=2, beta=1.e-3, rbf_lengthscale=.5,
        rbf_noise=1.e-2, step_margin_weight=100., seed=2,
    )
    checked = VIZ.validate_trace(metadata, events)
    assert len(checked) == 32
    assert metadata["diagnostic_only"] is True
    assert metadata["enters_replay"] is False
    assert metadata["acquisition"]["reference_rows"] == 0
    assert {row["episode_status"] for row in events[-16:]} == {"cutoff"}
    assert all(not row["committed_success"] for row in events)
    assert len(lambda_rows) == 32 * 4
    assert all(len(row["selected_ids"]) == 4 for row in events)
    assert all(len(row["all_K"]) == 16 for row in events)
    assert all(
        abs(row["acquisition"]["conditional_normalized_ess"][0] - 1.0) < 1.e-7
        for row in events
    )


def test_all_negative_B_terminates_each_lineage_nvp_without_fallback():
    metadata, events, lambda_rows = S.collect_trace(
        _Policy(), _Task(positive=False), gamma=.5, scenario_id=250_007,
        diagnostic_steps=3, beta=1.e-3, rbf_lengthscale=.5,
        rbf_noise=1.e-2, step_margin_weight=100., seed=2,
    )
    VIZ.validate_trace(metadata, events)
    assert len(events) == 16
    assert {row["episode_status"] for row in events} == {"nvp"}
    assert all(row["executed_id"] is None for row in events)
    assert len(lambda_rows) == 64
    assert not any(row["execution_eligible"] for row in lambda_rows)
