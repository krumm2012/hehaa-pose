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


def racket_physical_velocity(
    previous,
    current,
    previous_point,
    current_point,
    homography=None,
    height_m=None,
    camera_height_m=2.4,
):
    """Compute physical ground-plane or height-compensated speed (m/s and km/h) via calibrated Homography.

    Adheres strictly to the source-time measurement contract:
    - Same consecutive frame and time criteria as racket_image_velocity.
    - Requires non-singular 3x3 Homography projection to ground meters.
    - When height_m > 0, debiases perspective ground-dilation of elevated racket/ball.
    - Applies biomechanical centripetal acceleration ceiling to prevent detection jitter spikes.
    - Returns: (speed_mps, speed_kmh, status_or_basis)
      When uncalibrated or homography is absent, returns (None, None, 'uncalibrated').
    """
    if homography is None:
        return None, None, 'uncalibrated'
    if previous is None or previous_point is None or current_point is None:
        return None, None, 'missing_observation'
    t0, b0 = source_timestamp(previous)
    t1, b1 = source_timestamp(current)
    if t0 is None or t1 is None or b0 != b1 or t1 <= t0:
        return None, None, 'invalid_source_time'
    if b1 == 'nominal_fps':
        return None, None, 'estimated_time_not_a_speed_measurement'
    if current.get('frame_id', 0) - previous.get('frame_id', 0) != 1:
        return None, None, 'nonconsecutive_observations'

    from ground_reference import validate_homography_matrix
    valid_h, cond, det, reason = validate_homography_matrix(homography)
    if not valid_h:
        return None, None, f'degraded_ill_conditioned_homography: {reason}'

    h = height_m if height_m is not None else current.get('racket_height_m')
    try:
        from ground_reference import map_point, map_point_at_height
        if h is not None and float(h) > 0.0:
            p0_m = map_point_at_height(homography, previous_point, height_m=float(h), camera_height_m=camera_height_m)
            p1_m = map_point_at_height(homography, current_point, height_m=float(h), camera_height_m=camera_height_m)
            status = 'homography_height_debiased'
        else:
            p0_m = map_point(homography, previous_point)
            p1_m = map_point(homography, current_point)
            status = 'homography_ground_calibrated'
    except Exception as exc:
        return None, None, f'degraded_homography_mapping_failed: {exc}'

    if any(abs(v) > 100.0 for v in p0_m + p1_m):
        return None, None, f'degraded_projection_out_of_bounds: p0={p0_m}, p1={p1_m}'

    dist_m = math.dist(p0_m, p1_m)
    dt = t1 - t0
    mps = dist_m / dt
    if not math.isfinite(mps):
        return None, None, 'nonfinite_metric_velocity'

    if dist_m > 4.0 or mps > 90.0:
        return None, None, f'degraded_unphysical_displacement: dist={dist_m:.2f}m, speed={mps*3.6:.0f}km/h'

    from kinematic_smoothing import clamp_centripetal_speed
    mps_filtered, _ = clamp_centripetal_speed(mps)
    kmh = mps_filtered * 3.6
    return round(mps_filtered, 3), round(kmh, 2), status


