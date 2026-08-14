import numpy as np

import sfm_b1_store as S


GAMMAS = (.1, .2, .3, .4, .5, .7, 1.0)


def _result(y=1, full_h=True):
    return dict(
        resolved=True, y=int(y), taskspace=bool(y), collision_free=bool(y),
        certificate=bool(y), full_h=bool(full_h), terminal_step=(10 if full_h else 4),
        train_eligible=bool(y and full_h), segment=np.zeros((11, 2), np.float32),
        pedestrian_prediction=np.zeros((11, 1, 2), np.float32), diagnostics={},
    )


def _shard(round_i, counts=None):
    counts = counts or {gamma: 160 for gamma in GAMMAS}
    shard = S.RoundShard(round_i)
    for gamma in GAMMAS:
        context_id = shard.add_context(
            scenario_id=100 + int(gamma * 10), gamma=gamma, step=0,
            state=np.zeros(4), hp10=np.zeros((10, 16, 12)), low5=np.zeros(5),
            hist=np.zeros((16, 2)), ped_xy=np.zeros((1, 2)), ped_vel=np.zeros((1, 2)),
        )
        for index in range(counts[gamma]):
            shard.add_resolved_query(
                context_id, index, np.zeros((10, 2)), sigma=.5, result=_result(),
                acquisition_step=index % 4, gp_base_sigma=float(index),
            )
        # A resolved negative may never enter GP memory.
        shard.add_resolved_query(
            context_id, 10_000, np.zeros((10, 2)), sigma=.5, result=_result(y=0),
            acquisition_step=0, gp_base_sigma=1.e6,
        )
    return shard


def test_rotating_gamma_quotas_are_tight_over_two_rounds():
    first = S.gp_round_quotas(1, GAMMAS)
    second = S.gp_round_quotas(2, GAMMAS)
    assert sum(first.values()) == sum(second.values()) == 256
    assert set(first.values()) == set(second.values()) == {36, 37}
    combined = [first[gamma] + second[gamma] for gamma in sorted(first)]
    assert sum(combined) == 512
    assert min(combined) == 73 and max(combined) == 74


def test_gp_retention_uses_only_per_gamma_upper_quartile():
    shard = _shard(1)
    report = S.retain_gp_upper_quartile(shard, GAMMAS, seed=71)
    assert report["retained"] == report["requested"] == 256
    retained = [query for query in shard.queries if query["gp_retained"]]
    assert len(retained) == 256
    for query in retained:
        context = shard.contexts[query["context_id"]]
        diagnostic = report["per_gamma"][str(float(context["gamma"]))]
        assert query["y"] == 1 and query["full_h"]
        assert query["gp_base_sigma"] >= diagnostic["q75"]
    assert all(value["retained"] in (36, 37) for value in report["per_gamma"].values())


def test_gp_retention_does_not_redistribute_gamma_shortfall():
    counts = {gamma: 160 for gamma in GAMMAS}
    counts[.1] = 8
    shard = _shard(2, counts)
    report = S.retain_gp_upper_quartile(shard, GAMMAS, seed=72)
    low = report["per_gamma"]["0.1"]
    assert low["eligible"] == low["retained"] == 2
    assert low["shortfall"] == low["requested"] - 2
    assert report["retained"] == 256 - low["shortfall"]
    for gamma, value in report["per_gamma"].items():
        if gamma != "0.1":
            assert value["retained"] == value["requested"]


def test_recent_gp_records_are_only_last_two_retained_rounds(tmp_path):
    recent = S.RecentRounds(tmp_path, window=2)
    for round_i in (1, 2, 3):
        shard = _shard(round_i)
        S.retain_gp_upper_quartile(shard, GAMMAS, seed=80 + round_i)
        recent.append_and_save(shard)
    records = recent.gp_records()
    assert len(records) == 512
    assert {shard.round_i for shard, _ in records} == {2, 3}
    assert all(query["y"] == 1 and query["full_h"] for _, query in records)
