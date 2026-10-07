"""Optional planar diagnostics; no new inference, contact claim or scoring rule."""
from __future__ import annotations

import hashlib
import html
import json
import math
from collections import Counter
from statistics import median

import numpy as np

from observation_policy import finite_number, qualified_front_point

SCHEMA = 'tennis.ground-calibration.v1'
POLICY_VERSION = 'ground_reference_v1_original_pixels_ankle_candidates'
CORNER_IDS = ['A', 'B', 'C', 'D']
FEET = ('left_ankle', 'right_ankle')


def _size(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(v) is not int or v < 2 for v in value)):
        raise ValueError('image_size 必须是原图宽、高两个整数')
    return list(value)


def _point(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('坐标必须是两个有限数值')
    result = [finite_number(v) for v in value]
    if any(v is None for v in result):
        raise ValueError('坐标必须是两个有限数值')
    return result


def _positive(value, name):
    value = finite_number(value)
    if value is None or value <= 0:
        raise ValueError(f'{name} 必须为正有限数值')
    return value


def validate_homography_matrix(
    matrix: Any,
    max_condition_number: float = 1e7,
    min_abs_det: float = 1e-15,
) -> Tuple[bool, float, float, str]:
    """Validate 3x3 homography matrix for non-degeneracy and numerical stability.

    Returns:
        (is_valid, condition_number, determinant, status_message)
    """
    if matrix is None:
        return False, float("inf"), 0.0, "matrix_is_none"
    try:
        mat = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    except Exception:
        return False, float("inf"), 0.0, "invalid_shape"

    if not np.isfinite(mat).all():
        return False, float("inf"), 0.0, "non_finite_elements"

    try:
        u, s, vt = np.linalg.svd(mat)
        s_max = float(s[0])
        s_min = float(s[-1])
        if s_min < 1e-14:
            return False, float("inf"), 0.0, "near_zero_singular_value"
        cond = s_max / s_min
        det = float(np.linalg.det(mat))
    except Exception as exc:
        return False, float("inf"), 0.0, f"svd_failed: {exc}"

    if not math.isfinite(cond) or cond > max_condition_number:
        return False, cond, det, f"ill_conditioned_matrix (cond={cond:.1e} > {max_condition_number:.1e})"

    if not math.isfinite(det) or abs(det) < min_abs_det:
        return False, cond, det, f"singular_matrix (abs(det)={abs(det):.1e} < {min_abs_det:.1e})"

    return True, cond, det, "well_conditioned"


def map_point(matrix, point):
    homogeneous = np.asarray(matrix, dtype=float).reshape(3, 3) @ np.array([*point, 1.])
    denom = homogeneous[2]
    if not np.isfinite(homogeneous).all() or abs(denom) < 1e-4:
        raise ValueError('地面映射接近奇点')
    result = (homogeneous[:2] / denom).tolist()
    if not all(math.isfinite(v) for v in result):
        raise ValueError('地面映射非有限')
    return result


def map_point_at_height(matrix, point, height_m=0.0, camera_height_m=2.4, camera_center_xy=(0.0, 0.0)):
    """Map image point to physical world coordinates at given elevation height_m above ground plane.

    In perspective camera geometry, an elevated point at height h produces a ground-plane intercept
    dilated away from the camera optical projection center. This function cancels the perspective dilation:
    P = C_xy + (P_ground - C_xy) * (1.0 - h / H_cam).
    """
    ground_pt = map_point(matrix, point)
    h = float(height_m or 0.0)
    cam_h = float(camera_height_m or 2.4)
    if h <= 0.0 or cam_h <= h:
        return ground_pt
    scale = max(0.20, 1.0 - h / cam_h)
    cx, cy = camera_center_xy
    return [round(cx + (ground_pt[0] - cx) * scale, 4), round(cy + (ground_pt[1] - cy) * scale, 4)]



def fit_homography(points, world):
    # Normalize both coordinate domains; solve the 8 DoF system once at save/load.
    def normalize(p):
        p = np.asarray(p, dtype=float)
        center = p.min(axis=0)/2 + p.max(axis=0)/2
        distance = sum(math.hypot(*row)/len(p) for row in p-center)
        if not math.isfinite(distance) or distance <= np.finfo(float).tiny:
            raise ValueError('角点或实测尺寸无法稳定归一化')
        scale = math.sqrt(2) / distance
        transform = np.array([[scale, 0, -scale*center[0]],
                              [0, scale, -scale*center[1]], [0, 0, 1.]])
        return (p-center)*scale, transform
    src, ts = normalize(points)
    dst, td = normalize(world)
    a, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        a.extend([[x, y, 1, 0, 0, 0, -u*x, -u*y],
                  [0, 0, 0, x, y, 1, -v*x, -v*y]])
        b.extend([u, v])
    try:
        if not np.isfinite(a).all() or np.linalg.cond(a) > 1e10:
            raise ValueError('四点退化，无法建立稳定地面映射')
        h = np.append(np.linalg.solve(a, b), 1).reshape(3, 3)
        with np.errstate(over='ignore', invalid='ignore'):
            h = np.linalg.inv(td) @ h @ ts
    except np.linalg.LinAlgError as exc:
        raise ValueError('四点退化') from exc
    norm = math.hypot(*h.flat)
    if not np.isfinite(h).all() or not math.isfinite(norm) or norm == 0:
        raise ValueError('标定数值无法稳定映射')
    h = h / norm
    is_valid, cond, det, reason = validate_homography_matrix(h, max_condition_number=1e7)
    if not is_valid:
        raise ValueError(f'标定数值无法稳定映射: {reason}')
    return h.tolist()


def normalize_calibration(document):
    """Validate completely and recompute H; never trust an imported camera basis."""
    if not isinstance(document, dict) or document.get('schema') != SCHEMA:
        raise ValueError('不支持的地面标定格式')
    image_size = _size(document.get('image_size'))
    width = _positive(document.get('width_m'), '宽')
    length = _positive(document.get('length_m'), '长')
    flags = {}
    for key in ('dimensions_measured', 'camera_geometry_confirmed', 'correspondence_confirmed'):
        if type(document.get(key)) is not bool:
            raise ValueError(f'{key} 必须明确为布尔值')
        flags[key] = document[key]
    binding = document.get('binding')
    if (not isinstance(binding, dict) or binding.get('kind') not in ('video_sha256', 'camera_source')
            or not isinstance(binding.get('stream_id'), str)
            or not binding['stream_id']
            or not isinstance(binding.get('source_id'), str)
            or len(binding['source_id']) != 64
            or any(c not in '0123456789abcdef' for c in binding['source_id'])):
        raise ValueError('需要明确的机位/源内容绑定')
    binding = {k: binding[k] for k in ('kind', 'stream_id', 'source_id')}
    views = document.get('views')
    if not isinstance(views, dict) or 'front' not in views or set(views)-{'front', 'back'}:
        raise ValueError('必须提供正面四角；背面可选')
    world = [[0., 0.], [width, 0.], [width, length], [0., length]]
    normalized_views = {}
    for name, view in views.items():
        if (not isinstance(view, dict) or view.get('corner_ids') != CORNER_IDS
                or not isinstance(view.get('points'), list) or len(view['points']) != 4):
            raise ValueError('四角必须保留 A/B/C/D 实体对应，不自动排序')
        points = [_point(p) for p in view['points']]
        if any(not (0 <= x < image_size[0] and 0 <= y < image_size[1]) for x, y in points):
            raise ValueError('角点必须位于原图内')
        turns = []
        for i in range(4):
            p, q, r = points[i], points[(i+1) % 4], points[(i+2) % 4]
            turns.append((q[0]-p[0])*(r[1]-q[1])-(q[1]-p[1])*(r[0]-q[0]))
        if not (all(t > 1e-8 for t in turns) or all(t < -1e-8 for t in turns)):
            raise ValueError('四角不能重复、共线或交叉')
        matrix = fit_homography(points, world)
        denominators = [float(np.asarray(matrix)[2] @ [*p, 1.]) for p in points]
        if min(denominators) <= 0 <= max(denominators):
            raise ValueError('标定区域跨越映射奇点')
        normalized_views[name] = {'corner_ids': CORNER_IDS.copy(), 'points': points, 'H': matrix}
    checks = document.get('check_points', [])
    if not isinstance(checks, list):
        raise ValueError('独立检查点必须为列表')
    normalized_checks = []
    seen_checks = set()
    for check in checks:
        if not isinstance(check, dict) or check.get('view') not in normalized_views:
            raise ValueError('检查点视角无效')
        image = _point(check.get('image'))
        actual = _point(check.get('world_m'))
        identity = (check['view'], *image)
        if identity in seen_checks:
            raise ValueError('检查点重复')
        seen_checks.add(identity)
        if not (0 <= image[0] < image_size[0] and 0 <= image[1] < image_size[1]):
            raise ValueError('检查点必须在原图内')
        corners = normalized_views[check['view']]['points']
        if any(math.dist(image, p) < 1e-6 for p in corners):
            raise ValueError('拟合四角不能作为独立检查点')
        projected = map_point(normalized_views[check['view']]['H'], image)
        normalized_checks.append({'view': check['view'], 'image': image, 'world_m': actual,
                                  'error_m': math.dist(projected, actual)})
    result = {'schema': SCHEMA, 'policy_version': POLICY_VERSION,
              'coordinate_space': 'original_source_pixels', 'image_size': image_size,
              'binding': binding, 'width_m': width, 'length_m': length, **flags,
              'views': normalized_views, 'check_points': normalized_checks,
              'accuracy_validated': False, 'coaching_eligible': False,
              'contact_semantics': 'ankle_projection_not_observed_ground_contact'}
    origin = document.get('reference_origin')
    if origin is not None:
        if not isinstance(origin, dict) or origin.get('kind') not in ('manual_points', 'legacy_import', 'lane2_reference'):
            raise ValueError('标定参考来源无效')
        result['reference_origin'] = {k: str(origin[k])[:160] for k in
                                      ('kind', 'video_id', 'file_sha256', 'scale_note') if k in origin}
    # User-declared check points report residuals only; no automatic accuracy approval.
    fingerprint = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)
    result['calibration_id'] = hashlib.sha256(fingerprint.encode()).hexdigest()
    return result


class GroundReference:
    def __init__(self, calibration, binding, image_size, application=None):
        self.calibration = normalize_calibration(calibration)
        self.application = application
        self.reasons = []
        if self.calibration['binding'] != binding:
            self.reasons.append('source_binding_mismatch')
        if application is not None and application.get('input_binding') != binding:
            self.reasons.append('application_binding_mismatch')
        if self.calibration['image_size'] != list(image_size):
            self.reasons.append('source_image_size_mismatch')
        if not self.calibration['camera_geometry_confirmed']:
            self.reasons.append('camera_geometry_not_confirmed')
        # Cache the drawing geometry once; never recompute camera fitting per frame.
        self.overlay_geometry = {}
        if not self.reasons:
            for view, geometry in self.calibration['views'].items():
                inverse = np.linalg.inv(geometry['H'])
                width, length = self.calibration['width_m'], self.calibration['length_m']
                lines = []
                for i in range(11):
                    for pair in ([[width*i/10, 0], [width*i/10, length]],
                                 [[0, length*i/10], [width, length*i/10]]):
                        lines.append([map_point(inverse, p) for p in pair])
                self.overlay_geometry[view] = {'corners': geometry['points'], 'lines': lines}

    def observe(self, record):
        cal = self.calibration
        result = {'policy_version': POLICY_VERSION, 'calibration_id': cal['calibration_id'],
                  'source_frame_id': record.get('frame_id'), 'image_size': cal['image_size'],
                  'binding': cal['binding'], 'dimensions_measured': cal['dimensions_measured'],
                  'accuracy_validated': False, 'coaching_eligible': False,
                  'ground_contact_verified': False, 'views': {}, 'cross_view': {},
                  'reasons': self.reasons.copy()}
        if self.application is not None:
            result['application'] = self.application
        if self.reasons:
            return result
        if record.get('pose_observation_coordinate_space') != 'original_source_pixels':
            result['reasons'].append('observation_coordinate_space_unverified')
            return result
        frame_id = record.get('frame_id')
        observations = record.get('pose_observations')
        for view, geometry in cal['views'].items():
            raw = observations.get(view) if isinstance(observations, dict) else None
            rows = result['views'][view] = {}
            for foot in FEET:
                point = raw.get(foot) if isinstance(raw, dict) else None
                row = {'projected_xy_m': None, 'position_m': None, 'image_xy': None,
                       'ground_contact_verified': False, 'reasons': []}
                rows[foot] = row
                value, reason = qualified_front_point(point, frame_id, minimum_score=.5)
                if type(frame_id) is not int or frame_id < 0:
                    reason = 'invalid_source_frame_id'
                elif not isinstance(point, dict) or type(point.get('source_frame_id')) is not int or point['source_frame_id'] != frame_id:
                    reason = 'source_frame_unverified_or_mismatched'
                if reason:
                    row['reasons'].append(reason)
                    continue
                x, y, score = value
                if not (0 <= x < cal['image_size'][0] and 0 <= y < cal['image_size'][1]):
                    row['reasons'].append('point_outside_source_image')
                    continue
                # Reject extrapolation outside the explicitly marked plane patch.
                p = geometry['points']
                turns = [(p[(i+1)%4][0]-p[i][0])*(y-p[i][1])
                         -(p[(i+1)%4][1]-p[i][1])*(x-p[i][0]) for i in range(4)]
                if not (all(t >= -1e-7 for t in turns) or all(t <= 1e-7 for t in turns)):
                    row['reasons'].append('outside_calibrated_patch')
                    continue
                try:
                    row['projected_xy_m'] = map_point(geometry['H'], [x, y])
                except ValueError:
                    row['reasons'].append('projection_singularity')
                    continue
                row.update(image_xy=[x, y], model_score=score,
                           reasons=['ground_contact_not_observed'])
                if not cal['dimensions_measured']:
                    row['reasons'].append('dimensions_not_measured')
        for foot in FEET:
            a = result['views'].get('front', {}).get(foot, {}).get('projected_xy_m')
            b = result['views'].get('back', {}).get(foot, {}).get('projected_xy_m')
            delta = math.dist(a, b) if a is not None and b is not None else None
            reasons = ['ankle_projection_not_ground_contact_error']
            if delta is None: reasons.append('missing_qualified_view_pair')
            if not cal['correspondence_confirmed']:
                delta = None
                reasons.append('corner_correspondence_not_confirmed')
            result['cross_view'][foot] = {'projection_difference_m': delta,
                                          'accuracy_validated': False, 'reasons': reasons}
        return result


def event_ground_reference(event, frames):
    start, end = event.get('start_frame'), event.get('end_frame')
    if type(start) is not int or type(end) is not int or not 0 <= start <= end:
        return None
    rows = [f['ground_reference'] for f in frames if type(f.get('frame_id')) is int
            and start <= f['frame_id'] <= end
            and isinstance(f.get('ground_reference'), dict)
            and f['ground_reference'].get('source_frame_id') == f['frame_id']]
    if not rows:
        return None
    ids = {r.get('calibration_id') for r in rows}
    reasons = Counter(reason for r in rows for reason in r.get('reasons', []))
    result = {'policy_version': POLICY_VERSION, 'window': 'event_source_frame_interval',
              'calibration_ids': sorted(ids), 'frame_count': len(rows),
              'source_frame_ids': [r['source_frame_id'] for r in rows],
              'dimensions_measured': all(r.get('dimensions_measured') is True for r in rows),
              'accuracy_validated': False, 'coaching_eligible': False,
              'ground_contact_verified': False, 'feet': {}, 'rejected_reasons': dict(reasons)}
    applications = [r['application'] for r in rows if isinstance(r.get('application'), dict)]
    if applications:
        profiles = {a.get('profile_id') for a in applications}
        if len(profiles) != 1 or len(applications) != len(rows):
            result['rejected_reasons']['mixed_camera_profile_versions'] = len(rows)
            return result
        result['application'] = applications[0]
    if len(ids) != 1:
        result['rejected_reasons']['mixed_calibration_versions'] = len(rows)
        return result
    if len(set(result['source_frame_ids'])) != len(rows):
        result['rejected_reasons']['duplicate_source_frames'] = len(rows)
        return result
    for foot in FEET:
        pairs = [r['cross_view'][foot]['projection_difference_m'] for r in rows
                 if foot in r.get('cross_view', {})
                 and finite_number(r['cross_view'][foot].get('projection_difference_m')) is not None]
        missing = Counter(reason for r in rows for view in r.get('views', {}).values()
                          for reason in view.get(foot, {}).get('reasons', []))
        missing.update(reason for r in rows for reason in
                       r.get('cross_view', {}).get(foot, {}).get('reasons', []))
        result['feet'][foot] = {'paired_frames': len(pairs),
                               'median_projection_difference_m': median(pairs) if pairs else None,
                               'reasons': dict(missing)}
    return result


def ground_reference_html(event):
    reference = event.get('ground_reference')
    if not isinstance(reference, dict):
        return ''
    text = ['脚踝地面投影参考 · 着地未确认 · 不参与评分']
    application = reference.get('application') or {}
    if application.get('scope') == 'camera_profile':
        text.append('机位共享标定：'+str(application.get('camera_binding', {}).get('stream_id', '')))
    if not reference.get('dimensions_measured'):
        text.append('尺寸待实测')
    labels = {'camera_geometry_not_confirmed': '当前画面四角尚未核对',
              'source_binding_mismatch': '输入来源不匹配',
              'source_image_size_mismatch': '原图尺寸不匹配',
              'observation_coordinate_space_unverified': '观测坐标系未确认',
              'mixed_calibration_versions': '窗口含不同标定版本',
              'duplicate_source_frames': '窗口含重复源帧'}
    text.extend(labels[key] for key in reference.get('rejected_reasons', {}) if key in labels)
    for foot, name in zip(FEET, ('左脚踝', '右脚踝')):
        row = reference.get('feet', {}).get(foot, {})
        value = finite_number(row.get('median_projection_difference_m'))
        text.append(f'{name}双视角投影差 {value:.3f} m（{row.get("paired_frames", 0)}对）'
                    if value is not None else f'{name}缺少合格双视角配对')
        if 'corner_correspondence_not_confirmed' in row.get('reasons', {}):
            text.append('镜中四角实体对应尚未核对')
    return '<p class="ground-reference">' + html.escape('；'.join(text)) + '</p>'


def ground_reference_osd(reference):
    if not isinstance(reference, dict):
        return ''
    parts = ['Ground ref (ankle projections)']
    for foot, label in zip(FEET, ('L', 'R')):
        value = finite_number(reference.get('cross_view', {}).get(foot, {}).get('projection_difference_m'))
        parts.append(f'{label} delta {value:.3f}m' if value is not None else f'{label} delta N/A')
    parts.append('contact unknown')
    if not reference.get('dimensions_measured'):
        parts.append('size unverified')
    return ' | '.join(parts)
