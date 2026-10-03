"""Read persisted practice scores without silently migrating their policy or source."""

from typing import Dict

from practice_scoring import number, score_event


def resolve_practice_score(event: Dict) -> Dict:
    """Prefer a versioned score snapshot; compute only when none is usable."""
    saved = event.get("practice_score")
    if isinstance(saved, dict):
        value = saved.get("score")
        numeric = number(value)
        valid = (
            bool(saved.get("policy_version"))
            and saved.get("method") in {"coach_manual", "automatic_2d_projection"}
            and saved.get("scale") == [0, 100]
            and (value is None or (numeric is not None and 0 <= numeric <= 100))
        )
        if valid:
            result = dict(saved)
            # Manual-only exports need automatic evidence for legacy consumers,
            # but that evidence must never replace the saved manual score.
            if not isinstance(result.get("calibration"), dict):
                automatic = {k: v for k, v in event.items() if k != "practice_review"}
                result["calibration"] = score_event(automatic)["calibration"]
            return result
    return score_event(event)
