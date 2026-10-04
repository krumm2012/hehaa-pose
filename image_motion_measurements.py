"""Source-time image measurements, without guessed metric calibration."""
import math

POLICY_VERSION = 'image_motion_v1_source_time'


def source_timestamp(frame):
    source = frame.get('source_time')
    if source is not None:
        value = source.get('timestamp_seconds')
        basis = source.get('basis')
        if source.get('quality') not in ('reported', 'estimated') or basis not in ('media_pts', 'nominal_fps'):
            return None, basis
    else:
        value, basis = frame.get('timestamp'), 'legacy_unverified'
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None, basis
    return float(value), basis


def racket_image_velocity(previous, current, previous_point, current_point):
    """Fresh consecutive observations only; missing data stays missing."""
    if previous is None or previous_point is None or current_point is None:
        return None, 'missing_observation'
    t0, b0 = source_timestamp(previous)
    t1, b1 = source_timestamp(current)
    if t0 is None or t1 is None or b0 != b1 or t1 <= t0:
        return None, 'invalid_source_time'
    if current.get('frame_id', 0) - previous.get('frame_id', 0) != 1:
        return None, 'nonconsecutive_observations'
    speed = math.dist(previous_point, current_point) / (t1-t0)
    return (round(speed, 3), b1) if math.isfinite(speed) else (None, 'nonfinite_position')
