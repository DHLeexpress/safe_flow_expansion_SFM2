from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sfm_hp100_final_videos as V
import sfm_hp100_paper_video_style as STYLE


def test_fixed_world_frame_keeps_numeric_ticks_and_removes_axis_labels():
    figure, axis = plt.subplots()
    STYLE.fixed_world_frame(axis)
    assert axis.get_xlabel() == ""
    assert axis.get_ylabel() == ""
    assert tuple(axis.get_xlim()) == (-0.5, 6.5)
    assert tuple(axis.get_ylim()) == (-0.5, 6.5)
    assert axis.get_xticks().size > 0
    STYLE.fixed_world_frame(axis, bounds=(1.0, 3.0, 2.0, 5.0))
    assert tuple(axis.get_xlim()) == (1.0, 3.0)
    assert tuple(axis.get_ylim()) == (2.0, 5.0)
    plt.close(figure)


def test_safety_badge_is_optional_and_color_semantic():
    figure, axis = plt.subplots()
    STYLE.safety_badge(axis, True, enabled=False)
    assert not axis.texts
    STYLE.safety_badge(axis, True, enabled=True)
    assert len(axis.texts) == 1
    assert axis.texts[0].get_color() == STYLE.POSITIVE_BLUE
    axis.clear()
    STYLE.safety_badge(axis, False, enabled=True)
    assert axis.texts[0].get_color() == STYLE.NEGATIVE_RED
    plt.close(figure)


def test_all_B_uncertainty_colors_have_deterministic_order():
    attempt = {"selected_sigma": [0.2, 0.8, 0.1, 0.7, 0.4, 0.9]}
    np.testing.assert_array_equal(
        V._visible_uncertainty_indices(attempt), [5, 1, 3, 4, 0, 2],
    )


def test_acquisition_phases_hide_intermediate_families_and_spoiler_state():
    rows = [{
        "executed_group": "P2",
        "repair_attempts": [
            {"selected_local": None},
            {"selected_local": 3},
        ],
    }]
    phases = [(phase, attempt) for _, phase, attempt in V._acquisition_phases(rows)]
    assert phases == [
        ("raw", None),
        ("clear", None),
        ("uncertainty", 1),
        ("selected", 1),
        ("execute", 1),
    ]


def test_label_sheet_is_separate_from_scene_video(tmp_path):
    result = STYLE.render_label_sheet(tmp_path / "labels")
    assert Path(result["png"]).is_file()
    assert Path(result["pdf"]).is_file()


def test_manifest_requires_and_checks_mp4_sidecars(tmp_path):
    video = tmp_path / "case.mp4"
    video.write_bytes(b"video")
    STYLE.write_json(video.with_suffix(".json"), {
        "mp4_sha256": STYLE.sha256_file(video),
    })
    payload = V.build_manifest(tmp_path)
    assert payload["video_count"] == 1
    assert (tmp_path / "MANIFEST.json").is_file()
