"""Source-time image measurements, without guessed metric calibration."""
import math

POLICY_VERSION = 'image_motion_v2_source_identity'


def source_timestamp(frame):
    source = frame.get('source_time')
    if 'source_time' in frame:
        if not isinstance(source, dict):
            return None, 'invalid_source_time'
        value = source.get('timestamp_seconds')
        basis = source.get('basis')
        fid = frame.get('frame_id')
        if (source.get('schema_version') != 'tennis.source-time.v1'
                or type(fid) is not int or fid < 0
                or type(source.get('source_frame_id')) is not int
                or source['source_frame_id'] != fid
                or source.get('source_kind') != 'video_file'):
            return None, 'invalid_source_identity'
        if source.get('quality') not in ('reported', 'estimated') or basis not in ('media_pts', 'nominal_fps'):
            return None, basis
        if (basis, source.get('quality')) not in (('media_pts', 'reported'), ('nominal_fps', 'estimated')):
            return None, basis
    else:
        value, basis = frame.get('timestamp'), 'legacy_unverified'
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
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
    if b1 == 'nominal_fps':
        return None, 'estimated_time_not_a_speed_measurement'
    if current.get('frame_id', 0) - previous.get('frame_id', 0) != 1:
        return None, 'nonconsecutive_observations'
    speed = math.dist(previous_point, current_point) / (t1-t0)
    return (round(speed, 3), b1) if math.isfinite(speed) else (None, 'nonfinite_position')
