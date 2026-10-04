"""Expose stored event-window measurements without turning proxies into coaching scores."""
import math
import statistics
from collections import defaultdict

LABELS = {
    'hip_shoulder_separation': '肩髋分离角', 'shoulder_turn': '肩宽角度代理',
    'shoulder_turn_change': '肩部连线角度变化', 'preparation_knee_flexion': '准备期屈膝',
    'arm_extension': '持拍侧肘夹角', 'contact_lateral_distance': '触球横向距离',
    'weight_transfer': '躯干中心横移', 'balance_drift': '回位期躯干横移',
    'takeback_depth': '镜面手腕偏移比', 'scapular_retraction': '正背肩宽比',
    'racket_head_speed': '球拍框中心像素速度', 'brush_angle': '球拍像面轨迹倾角',
    'stance': '足部连线倾角', 'leg_drive': '髋部上移比例', 'kinematic_sequence': '动力链时序',
    'drop_depth_ratio': '最低点至触球候选上升比', 'racket_max_speed': '球拍框中心像素峰值',
}

METHOD_LABELS = {
    'image_plane_proxy_only': '像面投影', 'image_box_center_speed': '源时间 · 原始框中心位移（非拍头）', 'dual_view_anti_collapse': '双视角转体代理',
    'image_plane_change': '像面角度变化', 'image_plane_joint_angle': '二维关节角度',
    'swing_peak_window_2d': '挥拍峰值窗口', 'observable_2d': '二维观测',
    'screen_body_center_displacement': '画面身体中心位移',
    'dual_view_mirror_projection': '双视角镜面投影',
    'unvalidated_extended_proxy': '原始代理 · 有效性未确认',
}


def scoring_blockers(event):
    """Explain five-dimensional abstention without inventing rating rules."""
    from practice_scoring import POLICY, score_review
    from observation_policy import contact_evidence
    review = event.get('practice_review')
    if review and score_review(review)['status'] == 'coach_confirmed':
        return []
    contact = contact_evidence(event)
    status = event.get('contact_status', contact.get('contact_status'))
    metrics = (event.get('biomechanics') or {}).get('metrics') or {}
    required = {'preparation':['shoulder_turn_change','takeback_depth'],
                'positioning':['weight_transfer'], 'contact':['contact_lateral_distance'],
                'coordination':['kinematic_sequence','balance_drift'],
                'recovery':['balance_drift']}
    result = []
    for dimension in POLICY['dimensions']:
        reasons = ['automatic_rubric_not_independently_validated']
        if status != 'confirmed': reasons.append('contact_not_confirmed')
        if event.get('is_shadow_swing') or contact.get('is_shadow_swing'):
            reasons.append('shadow_swing')
        missing = [key for key in required.get(dimension['id'], []) if metrics.get(key, {}).get('value') is None]
        if missing: reasons.append('missing_observations')
        result.append({'dimension':dimension['id'], 'label':dimension['label'],
                       'score':None,'reasons':reasons,'missing_observations':missing,
                       'requirements_are_diagnostic_not_validated_rubric':True})
    return result


def _qualification(row, evidence):
    row['measurement_evidence'] = evidence or {}
    row['display_eligible'] = evidence.get('display_eligible') if evidence else None
    row['accuracy_validated'] = False
    row['qualification_status'] = ('image_observation_only' if row['display_eligible'] is True else
                                   'insufficient_evidence' if row['display_eligible'] is False else 'unvalidated_observation')
    return row


def event_analysis_metrics(event):
    rows = []
    for key, metric in ((event.get('biomechanics') or {}).get('metrics') or {}).items():
        if not isinstance(metric, dict):
            continue
        # Retired uncalibrated km/h values remain in historical files, not current summaries.
        if key in ("racket_head_speed", "racket_speed") and metric.get("unit") == "km/h":
            continue
        value = metric.get('value')
        if value is None or isinstance(value, (dict, list, bool)):
            continue
        if isinstance(value, (float, int)) and not math.isfinite(value):
            continue
        rows.append({'key': key, 'label': LABELS.get(key, key), 'value': value,
                     'unit': metric.get('unit') or '', 'confidence': metric.get('confidence'),
                     'coach_eligible': bool(metric.get('coach_eligible')),
                     'observability': metric.get('observability'),
                     'method_label': METHOD_LABELS.get(metric.get('observability'), '观测参考'),
                     'sample_count': metric.get('sample_count', (metric.get('details') or {}).get('sample_count')),
                     'source_frames': metric.get('source_frames') or [],
                     'contract': metric.get('contract')})
    # HUD uses extended raw observations even when the qualified metric abstains.
    # Expose those for evidence comparison, retaining abstention in the canonical metric.
    ext = (event.get('biomechanics') or {}).get('extended_biomechanics') or event.get('extended_biomechanics') or {}
    present = {r['key'] for r in rows}
    for key, section, field, unit in [
        ('racket_head_speed', 'racket_head_speed', 'contact_px_s', 'px/s'),
        ('racket_max_speed', 'racket_head_speed', 'max_px_s', 'px/s'),
        ('brush_angle', 'brush_angle', 'low_to_high_angle_deg', 'deg'),
        ('drop_depth_ratio', 'brush_angle', 'drop_depth_ratio', 'ratio'),
        ('leg_drive', 'leg_drive', 'drive_ratio', 'ratio'),
        ('stance', 'stance', 'image_foot_line_angle_deg', 'deg_2d'),
    ]:
        value = (ext.get(section) or {}).get(field)
        if key in present or value is None or isinstance(value, (dict, list, bool)):
            continue
        if isinstance(value, (int, float)) and not math.isfinite(value):
            continue
        rows.append({'key': key, 'label': LABELS[key], 'value': value, 'unit': unit,
                     'confidence': 0, 'coach_eligible': False,
                     'observability': 'unvalidated_extended_proxy',
                     'method_label': METHOD_LABELS['unvalidated_extended_proxy'],
                     'sample_count': None, 'source_frames': [],
                     'source': 'extended_biomechanics',
                     'contract': ((ext.get(section) or {}).get('metric_contracts') or {}).get(key)})
    for row in rows:
        evidence = (event.get('biomechanics') or {}).get('metrics', {}).get(row['key'], {}).get('measurement_evidence') or {}
        if row.get('source') == 'extended_biomechanics':
            section = {'drop_depth_ratio':'brush_angle','racket_max_speed':'racket_head_speed'}.get(row['key'],row['key'])
            evidence = dict((ext.get(section) or {}).get('measurement_evidence') or {})
            field = {'brush_angle':'low_to_high_angle_deg','drop_depth_ratio':'drop_depth_ratio',
                     'stance':'image_foot_line_angle_deg','leg_drive':'drive_ratio'}.get(row['key'])
            evidence.update((evidence.get('fields') or {}).get(field, {}))
        _qualification(row, evidence)
    return rows


def session_analysis_metrics(events):
    # Group by stroke, unit and observable: heterogeneous measurements are not averaged together.
    events = list(events)
    groups = defaultdict(list)
    for event in events:
        for row in event_analysis_metrics(event):
            groups[(event.get('stroke_type') or 'Swing', row['key'], row['unit'],
                    row['observability'])].append(row)
    result = []
    for (stroke, key, unit, observable), rows in groups.items():
        numbers = [r['value'] for r in rows if isinstance(r['value'], (int, float))]
        total = sum((e.get('stroke_type') or 'Swing') == stroke for e in events)
        result.append({'stroke_type': stroke, 'key': key, 'label': LABELS.get(key, key),
                       'unit': unit, 'observability': observable, 'count': len(rows),
                       'method_label': METHOD_LABELS.get(observable, '观测参考'),
                       'total_events': total,
                       'coach_eligible_count': sum(r['coach_eligible'] for r in rows),
                       'median': statistics.median(numbers) if numbers else None,
                       'min': min(numbers) if numbers else None, 'max': max(numbers) if numbers else None,
                       'categories': {v: sum(r['value'] == v for r in rows)
                                      for v in sorted({r['value'] for r in rows if isinstance(r['value'], str)})}})
    return result
