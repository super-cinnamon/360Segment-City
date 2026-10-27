from collections import defaultdict
from math import isfinite


class RecentAgentHistory:
    """Keep strong observations for a bounded number of video frames."""

    def __init__(self, max_age_frames=30):
        self.max_age_frames = max(0, int(max_age_frames))
        self._tracks = {}

    def match(self, segment, frame_number, score_floor):
        """Return recent strong-observation context for a matching candidate."""
        score = segment.get("score")
        if score is None or not isfinite(score) or score < score_floor:
            return None

        identity = segment.get("agent_identity")
        history = self._tracks.get(identity)
        if history is None:
            return None

        age_frames = frame_number - history["last_strong_frame"]
        if (
            age_frames < 0
            or age_frames > self.max_age_frames
            or segment.get("class_name") != history["class_name"]
            or segment.get("side") not in history["sides"]
        ):
            return None

        return {
            "matched_frame": frame_number,
            "last_strong_frame": history["last_strong_frame"],
            "last_strong_score": history["last_strong_score"],
            "last_depth": history["last_depth"],
            "age_frames": age_frames,
        }

    def observe_strong_segments(self, segments, frame_number, score_threshold):
        """Refresh history only from strong detections that survived depth pruning."""
        self._expire(frame_number)
        current_tracks = defaultdict(list)
        for segment in segments:
            score = segment.get("score")
            depth = segment.get("mode_depth")
            identity = segment.get("agent_identity")
            if (
                not identity
                or score is None
                or not isfinite(score)
                or score < score_threshold
                or depth is None
                or not isfinite(depth)
            ):
                continue

            current_tracks[identity].append(segment)

        for identity, observations in current_tracks.items():
            strongest = max(observations, key=lambda segment: segment["score"])
            self._tracks[identity] = {
                "class_name": strongest.get("class_name"),
                "sides": {segment.get("side") for segment in observations},
                "last_strong_frame": frame_number,
                "last_strong_score": float(strongest["score"]),
                "last_depth": float(strongest["mode_depth"]),
            }

    def _expire(self, frame_number):
        expired = [
            identity
            for identity, history in self._tracks.items()
            if frame_number - history["last_strong_frame"] > self.max_age_frames
        ]
        for identity in expired:
            del self._tracks[identity]


def restore_persistent_depth_pruned_agents(
    candidates_by_frame,
    kept_by_frame,
    keep_fraction=0.5,
):
    """Restore depth-pruned detections for identities kept in most frames."""
    if len(candidates_by_frame) != len(kept_by_frame):
        raise ValueError("Candidate and kept frame counts must match")

    frame_count = len(kept_by_frame)
    if frame_count == 0:
        return kept_by_frame

    identity_frames = defaultdict(set)
    for frame_index, frame_segments in enumerate(kept_by_frame):
        for segment in frame_segments:
            identity_frames[segment["agent_identity"]].add(frame_index)

    persistent_identities = {
        identity
        for identity, frame_indexes in identity_frames.items()
        if len(frame_indexes) / frame_count > keep_fraction
    }

    for frame_index, candidates in enumerate(candidates_by_frame):
        kept_segments = kept_by_frame[frame_index]
        kept_identities = {segment["agent_identity"] for segment in kept_segments}
        restored = [
            segment
            for segment in candidates
            if segment["agent_identity"] in persistent_identities
            and segment["agent_identity"] not in kept_identities
            and segment.get("mode_depth") is not None
            and isfinite(segment["mode_depth"])
        ]
        kept_segments.extend(restored)
        kept_segments.sort(key=lambda segment: segment["mode_depth"])

    return kept_by_frame