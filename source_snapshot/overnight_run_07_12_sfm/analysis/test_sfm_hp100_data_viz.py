import json
from pathlib import Path

import numpy as np
import pytest
import torch

import sfm_hp100_data_viz as V
import sfm_hp100_dynamics as D
import sfm_hp100_features as F
import sfm_scene as SS
import stage2_hp100_data as DATA


def _write_dataset(root: Path, *, corrupt_hp=False):
    gamma = 0.5
    episode = 7
    states = [np.zeros(4, np.float32)]
    actions = [np.array([0.4, 0.1], np.float32), np.array([0.2, -0.1], np.float32)]
    states.append(D.step_numpy(states[-1], actions[0]).astype(np.float32))
    ped0 = np.stack([
        np.array([1.0 + .16 * index, .45 + .12 * (index % 4)], np.float32)
        for index in range(20)
    ])
    peds = [ped0, ped0 + np.array([.01, -.005], np.float32)]
    velocities = np.tile(np.array([.1, -.05], np.float32), (20, 1))
    frames, geometries = [], []
    for state, ped_xy in zip(states, peds):
        obstacles = np.concatenate(
            (ped_xy, np.full((20, 1), SS.R_PED, np.float32)), axis=1
        )
        frame, geometry = F.hp100_frame(
            state[:2], obstacles, sensing=SS.R_SENSE, n_base=16,
            obstacle_velocities=velocities, robot_velocity=state[2:4],
            predict_gain=F.PREDICT_GAIN, predict_tau=F.PREDICT_TAU,
            return_geometry=True,
        )
        frames.append(frame)
        geometries.append(geometry)
    frames = np.stack(frames).astype(np.float32)
    if corrupt_hp:
        frames[1, 3, 9] += np.float32(.01)
    controls = np.stack(actions)
    U = np.zeros((2, 10, 2), np.float32)
    audits = [
        DATA.audit_weighted_plan(
            state, target,
            tuple(geometry[key] for key in ("A", "b", "ref", "margins")),
            gamma,
        )
        for state, target, geometry in zip(states, U, geometries)
    ]
    low5 = torch.zeros((2, 5))
    low5[:, 4] = gamma
    payload = dict(
        schema_version=V.EXPECTED_SCHEMA, success_only=True, gamma=gamma,
        n_traj=1, n_seeds=1, episode_start=7, episode_stop_exclusive=8,
        dynamics=D.contract(), hp=torch.from_numpy(frames),
        low5=low5, hist=torch.zeros((2, 16, 2)),
        U=torch.from_numpy(U), state=torch.from_numpy(np.stack(states)),
        ped_xy=torch.from_numpy(np.stack(peds)),
        ped_vel=torch.from_numpy(np.stack([velocities, velocities])),
        executed_action=torch.from_numpy(controls),
        episode=torch.tensor([episode, episode]), step=torch.tensor([0, 1]),
        target_eligible=torch.ones(2, dtype=torch.bool),
        target_reason_code=torch.zeros(2, dtype=torch.int8),
        plan_candidate_count=torch.full((2,), 2048, dtype=torch.int32),
        plan_accepted_count=torch.ones(2, dtype=torch.int32),
        plan_rejected_count=torch.full((2,), 2047, dtype=torch.int32),
        plan_weighted_h=torch.from_numpy(np.stack([row["h"] for row in audits])),
        plan_first_violation=torch.full((2,), -1, dtype=torch.int16),
    )
    filename = "sfm_hp100_windows_g0.5.pt"
    path = root / filename
    torch.save(payload, path)
    manifest = dict(
        status=V.EXPECTED_DATA_STATUS, schema_version=V.EXPECTED_SCHEMA,
        dynamics=D.contract(), source_hashes=DATA._source_hashes(),
        target_contract=DATA.target_contract(),
        feature=dict(
            shape=[32, 100], nominal_polytope_n_base=16, velocity_aware=False,
            current_position_tangent=True,
            predict_gain=F.PREDICT_GAIN, predict_tau=F.PREDICT_TAU,
            contract=F.contract(),
        ),
        environment=dict(
            n_ped=20, pedestrian_radius=SS.R_PED, sensing_radius=SS.R_SENSE,
            goal=np.asarray(SS.GOAL, float).tolist(),
            task_bounds=[float(SS.TASK_LO), float(SS.TASK_HI)],
        ),
        files=[dict(
            gamma=gamma, file=filename, sha256=V._sha256(path),
            bytes=path.stat().st_size, successful_episodes=[episode],
        )],
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    return gamma, episode


def test_episode_audit_proves_k16_geometry_is_not_32_raster_rays(tmp_path):
    gamma, episode = _write_dataset(tmp_path)
    manifest, rows, _ = V.load_episode(tmp_path, gamma, episode)
    validated = V.validate_episode(manifest, rows)
    audit = validated["audit"]
    assert audit["contexts"] == 2
    assert audit["all_hp_bitwise_equal"] is True
    assert audit["hp_max_abs_error"] == 0.0
    assert audit["rollout_bitwise_equal"] is True
    assert audit["artificial_outer_faces"] == 16
    assert audit["observation_angular_rays"] == 32
    assert validated["histories"].shape == (2, 10, 32, 100)
    np.testing.assert_array_equal(validated["histories"][0, 0], rows["hp"][0])
    np.testing.assert_array_equal(validated["histories"][1, 1], rows["hp"][0])


def test_episode_audit_fails_closed_on_changed_stored_hp(tmp_path):
    gamma, episode = _write_dataset(tmp_path, corrupt_hp=True)
    manifest, rows, _ = V.load_episode(tmp_path, gamma, episode)
    with pytest.raises(RuntimeError, match="not bitwise equal"):
        V.validate_episode(manifest, rows)


def test_loader_fails_closed_on_changed_feature_source(tmp_path):
    gamma, episode = _write_dataset(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source_hashes"]["features"]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="source hash differs at features"):
        V.load_episode(tmp_path, gamma, episode)


def test_renderer_writes_mp4_selected_png_pdf_and_contract(tmp_path):
    dataset = tmp_path / "data"
    dataset.mkdir()
    gamma, episode = _write_dataset(dataset)
    output = tmp_path / "render"
    report = V.render(
        dataset, gamma, episode, output, selected_step=1,
        frame_stride=1, fps=2, dpi=45,
    )
    assert report["status"] == V.STATUS
    assert report["geometry_observation_separation"] == {
        "nominal_artificial_outer_faces": 16,
        "observation_angular_rays": 32,
        "equal": False,
        "statement": "K=16 nominal support geometry is independent of 32 raster rays",
    }
    assert report["selected_frame"]["selected_step"] == 1
    assert report["selected_frame"]["rendered_frame_index"] == 1
    condition = report["selected_frame"]["geometry_condition"]
    assert condition["minimum_margin"] > 0.0
    assert condition["cancellation_condition"] >= 1.0
    for kind in ("mp4", "png", "pdf", "geometry_npz"):
        artifact = Path(report["outputs"][kind]["path"])
        assert artifact.is_file() and artifact.stat().st_size > 0
        assert V._sha256(artifact) == report["outputs"][kind]["sha256"]
    sidecar = np.load(report["outputs"]["geometry_npz"]["path"])
    assert sidecar["A"].shape[1:] == (2,)
    assert sidecar["stored_hp100"].shape == (32, 100)
    assert sidecar["weighted_plan_H10"].shape == (10, 2)
    disk = json.loads(Path(report["contract_path"]).read_text())
    assert disk["provenance_audit"]["all_hp_bitwise_equal"] is True
    assert disk["render"]["rendered_frames"] == 2


def test_counterfactual_rejects_dataset_that_is_already_current_tangent(tmp_path):
    gamma, episode = _write_dataset(tmp_path)
    manifest, rows, _ = V.load_episode(tmp_path, gamma, episode)
    with pytest.raises(ValueError, match="already uses current-position tangent"):
        V.counterfactual_no_retreat(manifest, rows)
