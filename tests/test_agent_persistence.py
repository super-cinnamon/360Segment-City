import numpy as np

from src.tasks.agent_persistence import RecentAgentHistory, restore_persistent_depth_pruned_agents
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


def test_recent_history_matches_at_limit_and_expires_after_it():
    history = RecentAgentHistory(max_age_frames=30)
    strong = {
        "agent_identity": "mask:front:car:1",
        "class_name": "car",
        "side": "front",
        "score": 0.8,
        "mode_depth": 0.3,
    }
    candidate = {**strong, "score": 0.3}
    history.observe_strong_segments([strong], 100, score_threshold=0.8)

    assert history.match(candidate, 130, score_floor=0.2)["age_frames"] == 30
    assert history.match(candidate, 131, score_floor=0.2) is None


def test_recent_history_requires_same_class_and_side():
    history = RecentAgentHistory(max_age_frames=30)
    strong = {
        "agent_identity": "mask:front:car:1",
        "class_name": "car",
        "side": "front",
        "score": 0.8,
        "mode_depth": 0.3,
    }
    history.observe_strong_segments([strong], 100, score_threshold=0.8)

    assert history.match({**strong, "side": "right", "score": 0.3}, 101, 0.2) is None
    assert history.match({**strong, "class_name": "truck", "score": 0.3}, 101, 0.2) is None


def test_recent_history_keeps_all_sides_seen_in_the_strong_frame():
    history = RecentAgentHistory(max_age_frames=30)
    front = {
        "agent_identity": "mask:front:car:1",
        "class_name": "car",
        "side": "front",
        "score": 0.8,
        "mode_depth": 0.3,
    }
    right = {**front, "side": "right", "score": 0.7}
    history.observe_strong_segments([front, right], 100, score_threshold=0.5)

    assert history.match({**right, "score": 0.3}, 101, score_floor=0.2) is not None


def test_weak_observation_does_not_refresh_recent_history():
    history = RecentAgentHistory(max_age_frames=30)
    strong = {
        "agent_identity": "mask:front:car:1",
        "class_name": "car",
        "side": "front",
        "score": 0.8,
        "mode_depth": 0.3,
    }
    weak = {**strong, "score": 0.3}
    history.observe_strong_segments([strong], 100, score_threshold=0.8)
    history.observe_strong_segments([weak], 110, score_threshold=0.8)

    assert history.match(weak, 111, score_floor=0.2)["last_strong_frame"] == 100


def test_tracker_does_not_match_across_camera_sides():
    tracker = ObjectIDTracker(iou_threshold=0.1)
    mask = np.ones((4, 4), dtype=bool)

    front_id = tracker.get_id(mask, "car", 0, side="front")
    right_id = tracker.get_id(mask, "car", 1, side="right")

    assert front_id != right_id