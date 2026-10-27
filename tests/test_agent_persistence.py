import numpy as np

from src.tasks.agent_persistence import restore_persistent_depth_pruned_agents
from src.tasks.id_tracker import ObjectIDTracker


def segment(identity, depth=0.2):
    return {"agent_identity": identity, "instance_id": 1, "mode_depth": depth}


def test_restores_identity_kept_in_more_than_half_the_frames():
    candidates = [
        [segment("mask:car:1", 0.4)],
        [segment("mask:car:1", 0.2)],
        [segment("mask:car:1", 0.2)],
        [segment("mask:car:1", 0.2)],
    ]
    kept = [[], [candidates[1][0]], [candidates[2][0]], [candidates[3][0]]]

    result = restore_persistent_depth_pruned_agents(candidates, kept)

    assert result[0] == candidates[0]


def test_does_not_restore_identity_kept_in_exactly_half_the_frames():
    candidates = [
        [segment("mask:car:1", 0.4)],
        [segment("mask:car:1", 0.2)],
        [segment("mask:car:1", 0.2)],
        [segment("mask:car:1", 0.4)],
    ]
    kept = [[], [candidates[1][0]], [candidates[2][0]], []]

    result = restore_persistent_depth_pruned_agents(candidates, kept)

    assert result[0] == []
    assert result[3] == []


def test_identity_sources_prevent_numeric_id_collisions():
    candidates = [
        [segment("mask:car:1", 0.4), segment("panorama:car:1", 0.4)],
        [segment("mask:car:1", 0.2), segment("panorama:car:1", 0.4)],
        [segment("mask:car:1", 0.2), segment("panorama:car:1", 0.4)],
    ]
    kept = [
        [],
        [candidates[1][0]],
        [candidates[2][0]],
    ]

    result = restore_persistent_depth_pruned_agents(candidates, kept)

    assert [item["agent_identity"] for item in result[0]] == ["mask:car:1"]


def test_restoration_does_not_duplicate_kept_identity_in_a_frame():
    item = segment("mask:car:1", 0.2)
    candidates = [[segment("mask:car:1", 0.4), item], [item]]
    kept = [[item], [item]]

    result = restore_persistent_depth_pruned_agents(candidates, kept)

    assert result[0] == [item]


def test_restoration_skips_non_finite_depth():
    candidates = [
        [segment("mask:car:1", float("nan"))],
        [segment("mask:car:1", 0.2)],
        [segment("mask:car:1", 0.2)],
    ]
    kept = [[], [candidates[1][0]], [candidates[2][0]]]

    result = restore_persistent_depth_pruned_agents(candidates, kept)

    assert result[0] == []


def test_mask_tracker_does_not_reuse_an_id_in_one_frame():
    tracker = ObjectIDTracker(iou_threshold=0.1)
    mask = np.ones((4, 4), dtype=bool)

    first_id = tracker.get_id(mask, "car", 0)
    second_id = tracker.get_id(mask, "car", 0)

    assert first_id != second_id