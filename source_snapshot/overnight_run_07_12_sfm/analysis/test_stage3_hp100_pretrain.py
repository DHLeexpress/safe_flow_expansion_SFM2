import json

import pytest
import torch

import sfm_hp100_dynamics as DYN
import stage3_hp100_pretrain as P


def _write_one_gamma_dataset(root):
    episodes = torch.arange(500, dtype=torch.int64)
    # Give episode zero a second step so newest-to-oldest indexing is observable.
    episodes = torch.cat((torch.tensor([0]), episodes))
    steps = torch.cat((torch.tensor([0, 1]), torch.zeros(499, dtype=torch.int64)))
    hp = torch.zeros(len(episodes), 32, 100)
    hp[0].fill_(10.0)
    hp[1].fill_(11.0)
    eligible = torch.ones(len(episodes), dtype=torch.bool)
    eligible[0] = False
    reason = torch.zeros(len(episodes), dtype=torch.int8)
    reason[0] = P.DATA.TARGET_ALL_REJECTED
    accepted = torch.ones(len(episodes), dtype=torch.int32)
    accepted[0] = 0
    rejected = 2048 - accepted
    payload = {
        "schema_version": P.SCHEMA_VERSION,
        "success_only": True,
        "gamma": 0.1,
        "n_traj": 500,
        "dynamics": DYN.contract(),
        "hp": hp,
        "low5": torch.zeros(len(episodes), 5),
        "hist": torch.zeros(len(episodes), 16, 2),
        "U": torch.zeros(len(episodes), 10, 2),
        "episode": episodes,
        "step": steps,
        "plan_weighted_h": torch.ones(len(episodes), 11),
        "target_eligible": eligible,
        "target_reason_code": reason,
        "plan_candidate_count": torch.full((len(episodes),), 2048, dtype=torch.int32),
        "plan_accepted_count": accepted,
        "plan_rejected_count": rejected,
        "plan_first_violation": torch.full((len(episodes),), -1, dtype=torch.int16),
        "action_mean_max_abs_error": torch.zeros(len(episodes)),
        "target_contract": P.DATA.target_contract(),
        "eligible_windows": int(eligible.sum()),
        "excluded_all_rejected_windows": 1,
        "excluded_weighted_h10_failed_windows": 0,
    }
    path = root / "sfm_hp100_windows_g0.1.pt"
    torch.save(payload, path)
    manifest = {
        "status": "HP100_ID_DATASET_COMPLETE",
        "schema_version": P.SCHEMA_VERSION,
        "files": [{
            "gamma": 0.1,
            "file": path.name,
            "sha256": P.sha256_file(path),
        }],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_loader_is_exact_500_lineage_disjoint_and_lazy_hp10(tmp_path):
    _write_one_gamma_dataset(tmp_path)
    manifest_sha = P.sha256_file(tmp_path / "manifest.json")
    train, val, metadata = P.load_split(
        tmp_path, gammas=(0.1,), val_frac=0.1, seed=20260720,
        expected_manifest_sha256=manifest_sha, require_canonical=False,
    )
    train_episodes = set(train.episodes.tolist())
    val_episodes = set(val.episodes.tolist())
    assert train_episodes.isdisjoint(val_episodes)
    assert len(train_episodes) == 450
    assert len(val_episodes) == 50
    assert metadata["files"]["0.1"]["train_lineages"] == 450
    assert metadata["files"]["0.1"]["val_lineages"] == 50

    source = train.sources[0]
    assert source["hp"].shape == (501, 32, 100)
    assert source["_history_indices"].shape == (501, 10)
    assert source["_history_indices"][1].tolist() == [1] + [0] * 9
    # The only ten-frame tensor is gathered for one requested row.
    dataset = train if 0 in train_episodes else val
    position = torch.nonzero(dataset.episodes == 0, as_tuple=False).flatten()[-1]
    hp10 = dataset[int(position)][0]
    assert hp10.shape == (10, 32, 100)
    assert float(hp10[0, 0, 0]) == 11.0
    assert float(hp10[1, 0, 0]) == 10.0
    assert len(train) + len(val) == 500
    assert not hasattr(dataset, "hp10")


def test_loader_uses_one_validation_scenario_bank_across_gammas(tmp_path):
    _write_one_gamma_dataset(tmp_path)
    first_path = tmp_path / "sfm_hp100_windows_g0.1.pt"
    second_path = tmp_path / "sfm_hp100_windows_g0.2.pt"
    second = torch.load(first_path, weights_only=False)
    second["gamma"] = 0.2
    second["episode"] = second["episode"] + 25
    torch.save(second, second_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["files"].append({
        "gamma": 0.2, "file": second_path.name,
        "sha256": P.sha256_file(second_path),
    })
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    train, val, metadata = P.load_split(
        tmp_path, gammas=(0.1, 0.2),
        expected_manifest_sha256=P.sha256_file(tmp_path / "manifest.json"),
        require_canonical=False,
    )
    shared = set(metadata["shared_validation_episodes"])
    assert len(shared) == 50 and shared <= set(range(25, 500))
    assert set(val.episodes.tolist()) == shared
    assert shared.isdisjoint(set(train.episodes.tolist()))
    assert metadata["files"]["0.1"]["val_episodes"] == sorted(shared)
    assert metadata["files"]["0.2"]["val_episodes"] == sorted(shared)


def test_loader_rejects_less_than_500_successful_lineages(tmp_path):
    _write_one_gamma_dataset(tmp_path)
    path = tmp_path / "sfm_hp100_windows_g0.1.pt"
    payload = torch.load(path, weights_only=False)
    payload["n_traj"] = 499
    torch.save(payload, path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["files"][0]["sha256"] = P.sha256_file(path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    try:
        P.load_split(
            tmp_path, gammas=(0.1,),
            expected_manifest_sha256=P.sha256_file(tmp_path / "manifest.json"),
            require_canonical=False,
        )
    except ValueError as error:
        assert "exactly 500 successful lineages" in str(error)
    else:
        raise AssertionError("incomplete successful-lineage dataset was accepted")


def test_hierarchical_mass_is_gamma_trajectory_window():
    episodes = torch.tensor([1, 1, 2, 2, 2, 2, 7, 8, 8, 8])
    gammas = torch.tensor([0] * 6 + [1] * 4)
    weights = P.hierarchical_sampler_weights(episodes, gammas)
    assert torch.isclose(weights.sum(), torch.tensor(1.0, dtype=weights.dtype))
    assert torch.isclose(weights[gammas == 0].sum(), torch.tensor(0.5, dtype=weights.dtype))
    assert torch.isclose(weights[episodes == 1].sum(), torch.tensor(0.25, dtype=weights.dtype))
    assert torch.isclose(weights[episodes == 2].sum(), torch.tensor(0.25, dtype=weights.dtype))


class _ValidationPolicy(torch.nn.Module):
    u_max = 2.0
    d = 20

    def ctx_from(self, hp, low, hist):
        return low

    def forward(self, x, tau, ctx):
        return 0.1 * x + 0.0 * ctx[:, :1]


def test_validation_cfm_is_fixed_and_preserves_rng():
    count = 6
    source = {
        "hp": torch.zeros(count, 32, 100),
        "low5": torch.zeros(count, 5),
        "hist": torch.zeros(count, 16, 2),
        "U": torch.zeros(count, 10, 2),
        "episode": torch.arange(count),
        "_history_indices": torch.arange(count)[:, None].expand(-1, 10).clone(),
    }
    dataset = P.HP100WindowDataset(
        [source, source],
        gamma_rows=torch.tensor([0, 0, 0, 1, 1, 1]),
        source_rows=torch.tensor([0, 1, 2, 3, 4, 5]),
    )
    torch.manual_seed(9)
    before = torch.random.get_rng_state().clone()
    first = P.deterministic_validation(
        _ValidationPolicy(), dataset, (0.1, 0.2), "cpu", batch=2, seed=17
    )
    after = torch.random.get_rng_state().clone()
    second = P.deterministic_validation(
        _ValidationPolicy(), dataset, (0.1, 0.2), "cpu", batch=2, seed=17
    )
    assert first == second
    assert torch.equal(before, after)


class _MacroValidationPolicy(_ValidationPolicy):
    def forward(self, x, tau, ctx):
        target = -x / (1.0 - tau[:, None])
        return target + ctx[:, :1]


def test_validation_macro_matches_gamma_lineage_window_mass():
    def source(episodes, offsets):
        count = len(episodes)
        low = torch.zeros(count, 5)
        low[:, 0] = torch.as_tensor(offsets)
        return {
            "hp": torch.zeros(count, 32, 100), "low5": low,
            "hist": torch.zeros(count, 16, 2), "U": torch.zeros(count, 10, 2),
            "episode": torch.as_tensor(episodes),
            "_history_indices": torch.arange(count)[:, None].expand(-1, 10).clone(),
        }

    dataset = P.HP100WindowDataset(
        [source([1, 1, 2], [0.0, 0.0, 2.0]), source([7], [8.0 ** 0.5])],
        gamma_rows=torch.tensor([0, 0, 0, 1]),
        source_rows=torch.tensor([0, 1, 2, 0]),
    )
    macro, per_gamma = P.deterministic_validation(
        _MacroValidationPolicy(), dataset, (0.1, 0.2), "cpu", batch=2, seed=17,
    )
    assert abs(per_gamma["0.1"] - 2.0) < 1.0e-5
    assert abs(per_gamma["0.2"] - 8.0) < 1.0e-5
    assert abs(macro - 5.0) < 1.0e-5


def test_loader_requires_exact_manifest_sha(tmp_path):
    _write_one_gamma_dataset(tmp_path)
    try:
        P.load_split(
            tmp_path, gammas=(0.1,), expected_manifest_sha256="0" * 64,
            require_canonical=False,
        )
    except RuntimeError as error:
        assert "dataset manifest" in str(error)
    else:
        raise AssertionError("wrong expected manifest SHA was accepted")


def test_canonical_manifest_contract_accepts_only_declared_collection(tmp_path):
    source_git = {"root": "/repo", "head": "abc", "clean": True, "status": ""}
    files = []
    ranges = {}
    for gamma in map(float, P.SP.GAMMAS):
        progress = tmp_path / f"progress_{gamma}.json"
        data_file = f"data_{gamma}.pt"
        data_sha = f"sha-{gamma}"
        progress.write_text(json.dumps({
            "status": "HP100_GAMMA_COLLECTION_COMPLETE",
            "accepted_successes": P.DATA.SUCCESSES_PER_GAMMA,
            "target_successes": P.DATA.SUCCESSES_PER_GAMMA,
            "attempted_episodes": P.DATA.SUCCESSES_PER_GAMMA,
            "episode_range": [0, P.DATA.SUCCESSES_PER_GAMMA],
            "data_file": data_file, "data_sha256": data_sha,
            "windows": 600, "eligible_windows": 500,
            "excluded_all_rejected_windows": 80,
            "excluded_weighted_h10_failed_windows": 20,
        }))
        files.append({
            "gamma": gamma, "n_traj": P.DATA.SUCCESSES_PER_GAMMA,
            "file": data_file, "sha256": data_sha,
            "successful_episodes": list(range(P.DATA.SUCCESSES_PER_GAMMA)),
            "rejected_episodes": [],
            "attempted_episodes": P.DATA.SUCCESSES_PER_GAMMA,
            "episode_range": [0, P.DATA.SUCCESSES_PER_GAMMA],
            "progress_file": progress.name,
            "progress_sha256": P.sha256_file(progress),
            "windows": 600, "eligible_windows": 500,
            "excluded_all_rejected_windows": 80,
            "excluded_weighted_h10_failed_windows": 20,
            "runtime": {"requested_device": "cpu"},
        })
        ranges[str(gamma)] = [0, P.DATA.SUCCESSES_PER_GAMMA]
    manifest = {
        "canonical_full_run": True,
        "role": "successful SafeMPPI ID demonstrations for fresh Hp100 pretraining",
        "total_successful_lineages": P.DATA.SUCCESSES_PER_GAMMA * len(P.SP.GAMMAS),
        "total_context_rows": 600 * len(P.SP.GAMMAS),
        "total_eligible_windows": 500 * len(P.SP.GAMMAS),
        "target_contract": P.DATA.target_contract(),
        "episode_allocation": {
            "start": P.DATA.EPISODE_START,
            "successful_trajectories_per_gamma": P.DATA.SUCCESSES_PER_GAMMA,
            "max_attempts_per_gamma": P.DATA.MAX_ATTEMPTS_PER_GAMMA,
            "terminal_ranges": ranges,
        },
        "environment": {
            "n_ped": P.DATA.N_PED,
            "ped_speed_range": list(P.DATA.PED_SPEED_RANGE),
            "gammas": list(map(float, P.SP.GAMMAS)),
            "horizon": P.DATA.HORIZON, "T": P.DATA.T,
            "goal": torch.as_tensor(P.SS.GOAL).tolist(),
            "task_bounds": [float(P.SS.TASK_LO), float(P.SS.TASK_HI)],
            "pedestrian_radius": float(P.SS.R_PED),
            "sensing_radius": float(P.SS.R_SENSE),
        },
        "expert": {
            "name": P.DATA.HP100_EXPERT_NAME,
            "config": P.DATA.locked_expert_config(),
            "execution": (
                "CappedSafeMPPIAdapter using sfm_hp100_dynamics for internal and real "
                "steps; current-position tangent nominal geometry (predict_gain=0)"
            ),
            "supervised_target": (
                P.DATA.SUPERVISED_TARGET
            ),
        },
        "feature": {
            "shape": [32, 100], "contract": P.HPF.contract(),
            "dtype": "float32", "radial_bins": 100, "angular_bins": 32,
            "radial_pooling": "none",
            "nominal_polytope_n_base": 16, "velocity_aware": False,
            "current_position_tangent": True,
            "predict_gain": 0.0, "predict_tau": P.HPF.PREDICT_TAU,
        },
        "dynamics": P.DYN.contract(), "source_git": source_git,
        "source_completion_audit": {
            "git": source_git, "source_hashes_equal": True,
        },
        "source_hashes": P.DATA._source_hashes(),
        "runtime": {"python": "test"}, "files": files,
        "parallelism": {
            "gamma_device_map": {str(gamma): "cpu" for gamma in map(float, P.SP.GAMMAS)},
        },
    }
    P._validate_canonical_manifest(manifest, tmp_path)
    manifest["feature"]["nominal_polytope_n_base"] = 32
    try:
        P._validate_canonical_manifest(manifest, tmp_path)
    except ValueError as error:
        assert "nominal_polytope_n_base" in str(error)
    else:
        raise AssertionError("noncanonical 32-face nominal polytope was accepted")
    manifest["feature"]["nominal_polytope_n_base"] = 16
    manifest["feature"]["predict_gain"] = 0.25
    with pytest.raises(ValueError, match="feature.predict_gain"):
        P._validate_canonical_manifest(manifest, tmp_path)


def test_tensor_lineages_are_bound_to_manifest_ledger(tmp_path):
    path = tmp_path / "data.pt"
    path.write_bytes(b"payload")
    payload = {
        "episode": torch.tensor([500, 501]), "n_seeds": 2,
        "episode_start": 0, "episode_stop_exclusive": 2,
    }
    row = {
        "successful_episodes": [0, 1], "attempted_episodes": 2,
        "episode_range": [0, 2], "windows": 2, "bytes": path.stat().st_size,
    }
    try:
        P._validate_source_manifest_binding(payload, row, path, 0.1)
    except RuntimeError as error:
        assert "lineage IDs" in str(error)
    else:
        raise AssertionError("manifest/tensor lineage substitution was accepted")


def test_restart_only_outputs_refuse_existing_paths(tmp_path):
    outdir = tmp_path / "run"
    policy = tmp_path / "policy.pt"
    out, promoted = P._prepare_restart_only_outputs(outdir, policy)
    assert out == outdir.resolve() and promoted == policy.resolve()
    try:
        P._prepare_restart_only_outputs(outdir, tmp_path / "second.pt")
    except FileExistsError as error:
        assert "restart-only" in str(error)
    else:
        raise AssertionError("existing run directory was accepted")


def test_atomic_policy_publication_never_overwrites_racing_target(tmp_path, monkeypatch):
    target = tmp_path / "policy.pt"

    def racing_save(_policy, temporary, *, extra):
        del extra
        torch.save({"ours": True}, temporary)
        target.write_text("sentinel")

    monkeypatch.setattr(P.GPS, "save_sfm_hp100_policy", racing_save)
    try:
        P.atomic_policy_save(object(), target, extra={})
    except FileExistsError:
        pass
    else:
        raise AssertionError("racing promoted policy was overwritten")
    assert target.read_text() == "sentinel"
    assert not target.with_suffix(".pt.tmp").exists()


def test_authenticated_checkpoint_rejects_midload_mutation(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"before")

    def mutating_load(checkpoint, *, device):
        del device
        checkpoint.write_bytes(b"after")
        return object(), {}

    monkeypatch.setattr(P.GPS, "load_sfm_hp100_policy", mutating_load)
    try:
        P.load_authenticated_checkpoint(path, device="cpu")
    except RuntimeError as error:
        assert "changed while" in str(error)
    else:
        raise AssertionError("mid-load checkpoint mutation was accepted")


def test_promotion_gate_rejects_non_id_or_non_unit_temperature():
    good = {
        "distribution": "ID",
        "temperature": 1.0,
        "per_gamma": {"0.1": {
            "SR": 0.8, "CR": 0.2, "timeout": 0.0, "Validity": 0.7,
            "successful_clearance": 0.1, "successful_time_to_goal": 8.0,
        }},
    }
    good.update(M_per_gamma=10, ep0=12000, noise_seed=17)
    P._validate_gate(good, (0.1,), M=10, ep0=12000, noise_seed=17)
    for key, value in (("distribution", "OOD"), ("temperature", 0.5)):
        bad = dict(good)
        bad[key] = value
        try:
            P._validate_gate(bad, (0.1,), M=10, ep0=12000, noise_seed=17)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"invalid promotion gate accepted: {key}={value}")
    for key, value in (("M_per_gamma", 9), ("ep0", 14000), ("noise_seed", 18)):
        bad = dict(good)
        bad[key] = value
        try:
            P._validate_gate(bad, (0.1,), M=10, ep0=12000, noise_seed=17)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"changed promotion bank accepted: {key}={value}")


def test_warmup_cosine_does_not_zero_the_last_training_epoch():
    assert P._lr_multiplier(0, 120, 5) == 0.2
    assert P._lr_multiplier(4, 120, 5) == 1.0
    assert P._lr_multiplier(119, 120, 5) > 0.0
    assert P._lr_multiplier(120, 120, 5) == 0.0
