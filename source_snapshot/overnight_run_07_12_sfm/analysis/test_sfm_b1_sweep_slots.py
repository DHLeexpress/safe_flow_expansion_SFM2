import sfm_b1_sweep as SW


def test_cpu_pools_keep_legacy_slots_and_add_eight_disjoint_subslots(tmp_path):
    cpulist = tmp_path / "cpulist"
    cpulist.write_text("64-127\n")

    pools = SW.cpu_pools(cpulist)

    assert pools["1"] == list(range(64, 96))
    assert pools["3"] == list(range(96, 128))
    expected = {
        "1a": range(64, 72), "1b": range(72, 80),
        "1c": range(80, 88), "1d": range(88, 96),
        "3a": range(96, 104), "3b": range(104, 112),
        "3c": range(112, 120), "3d": range(120, 128),
    }
    for slot, values in expected.items():
        assert pools[slot] == list(values)
    assert len(set().union(*(set(pools[slot]) for slot in expected))) == 64
    assert sorted(sum((pools[slot] for slot in ("1a", "1b", "1c", "1d")), [])) == pools["1"]
    assert sorted(sum((pools[slot] for slot in ("3a", "3b", "3c", "3d")), [])) == pools["3"]


def test_gpu_slot_mapping_reuses_the_physical_gpu_uuid():
    expected = {
        "1": "1", "1a": "1", "1b": "1", "1c": "1", "1d": "1",
        "3": "3", "3a": "3", "3b": "3", "3c": "3", "3d": "3",
    }
    for slot, gpu_index in expected.items():
        assert SW.gpu_index_for_slot(slot) == gpu_index
        assert SW.GPU_UUIDS[SW.gpu_index_for_slot(slot)] == SW.GPU_UUIDS[gpu_index]


def test_run_parallel_combines_subslot_cpu_pool_with_physical_gpu(monkeypatch, tmp_path):
    pools = {
        "1a": list(range(8)), "1b": list(range(8, 16)),
        "1c": list(range(16, 24)), "1d": list(range(24, 32)),
        "3a": list(range(32, 40)), "3b": list(range(40, 48)),
        "3c": list(range(48, 56)), "3d": list(range(56, 64)),
    }
    calls = []

    class CompletedProcess:
        @staticmethod
        def wait():
            return 0

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return CompletedProcess()

    monkeypatch.setattr(SW, "cpu_pools", lambda: pools)
    monkeypatch.setattr(SW, "_job_environment", lambda uuid: {"gpu_uuid": uuid})
    monkeypatch.setattr(SW.subprocess, "Popen", fake_popen)

    logs = SW.run_parallel([
        ("1a", ["python", "a.py"], "one_a"),
        ("1b", ["python", "b.py"], "one_b"),
        ("1c", ["python", "e.py"], "one_c"),
        ("1d", ["python", "f.py"], "one_d"),
        ("3a", ["python", "c.py"], "three_a"),
        ("3b", ["python", "d.py"], "three_b"),
        ("3c", ["python", "g.py"], "three_c"),
        ("3d", ["python", "h.py"], "three_d"),
    ], tmp_path)

    assert [call[0][2] for call in calls] == [
        SW._compact_cpu_list(pools[slot])
        for slot in ("1a", "1b", "1c", "1d", "3a", "3b", "3c", "3d")
    ]
    assert [call[1]["gpu_uuid"] for call in calls] == [
        *([SW.GPU_UUIDS["1"]] * 4), *([SW.GPU_UUIDS["3"]] * 4),
    ]
    assert logs == [
        str(tmp_path / "one_a.log"), str(tmp_path / "one_b.log"),
        str(tmp_path / "one_c.log"), str(tmp_path / "one_d.log"),
        str(tmp_path / "three_a.log"), str(tmp_path / "three_b.log"),
        str(tmp_path / "three_c.log"), str(tmp_path / "three_d.log"),
    ]


def test_unknown_gpu_slot_is_rejected():
    try:
        SW.gpu_index_for_slot("2a")
    except ValueError as error:
        assert "unknown GPU/CPU slot" in str(error)
    else:
        raise AssertionError("unknown slot was accepted")
