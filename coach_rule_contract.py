"""Code-owned approval for independently validated automatic teaching rules.

Observation eligibility, a model confidence, historical labels, and manual
scores cannot approve an automatic rule. Keep capture/review guidance available.
"""
POLICY_VERSION = 'automatic_coach_v2_validated_rules_only'
EXCLUSION_REASON = 'automatic_technique_rule_not_independently_validated'
VALIDATED_TECHNIQUE_RULES = frozenset()

# Pure 2D image-plane & dual-view observable rules (7-point racket + shoulder-spine ratio)
OBSERVABLE_2D_TECHNIQUE_RULES = frozenset({
    'contact_too_close',
    'limited_arm_extension',
    'limited_knee_flexion',
    'limited_brush_drop',
    'limited_shoulder_turn',
    'disconnected_kinetic_chain',
    'insufficient_whip_release',
})

OBSERVABLE_2D_THRESHOLDS = {
    'min_arm_extension_deg': 135.0,
    'good_arm_extension_deg': 148.0,
    'min_brush_angle_deg': 14.0,
    'min_drop_depth_ratio': 0.25,
    'min_contact_lateral_bw': 0.55,
    'min_shoulder_spine_delta': 0.20,
    'min_dual_view_turn_span_deg': 25.0,
    'min_whip_amplification_ratio': 2.20,
}


def technique_rule_approved(rule_id, enable_2d_observable=False):
    if enable_2d_observable and rule_id in OBSERVABLE_2D_TECHNIQUE_RULES:
        return True
    return rule_id in VALIDATED_TECHNIQUE_RULES


def automatic_coach_policy(enable_2d_observable=False):
    active = OBSERVABLE_2D_TECHNIQUE_RULES if enable_2d_observable else VALIDATED_TECHNIQUE_RULES
    return {'policy_version': 'observable_2d_dual_view_v1' if enable_2d_observable else POLICY_VERSION,
            'validated_rule_ids': sorted(active),
            'approval_source': 'dual_view_2d_observable_contract' if enable_2d_observable else 'code_owned_independent_validation',
            'exclusion_reason': None if enable_2d_observable else EXCLUSION_REASON,
            'accuracy_validated': False}


def evaluate_2d_observable_diagnostics(summary):
    """Evaluate the 5 pure 2D observable biomechanical rules on a swing event summary."""
    t = OBSERVABLE_2D_THRESHOLDS
    rules = []

    # 1. Arm extension at contact (2D elbow angle)
    elbow_deg = summary.get('contact_elbow_deg')
    if elbow_deg is not None:
        ok = elbow_deg >= t['min_arm_extension_deg']
        good = elbow_deg >= t['good_arm_extension_deg']
        rules.append({
            'rule_id': 'limited_arm_extension',
            'title': '触球持拍臂延展度',
            'status': 'pass' if good else ('ok' if ok else 'warn'),
            'value': round(float(elbow_deg), 1),
            'unit': '°',
            'target': f"≥{t['min_arm_extension_deg']:.0f}°",
            'cue': '触球手臂舒展充分' if ok else '挥拍触球时手臂再舒展',
        })

    # 2. 7-point sweet_spot low-to-high brush angle & drop depth
    brush_deg = summary.get('brush_angle_deg')
    drop_ratio = summary.get('drop_depth_ratio')
    if brush_deg is not None or drop_ratio is not None:
        ok_b = brush_deg is not None and brush_deg >= t['min_brush_angle_deg']
        ok_d = drop_ratio is not None and drop_ratio >= t['min_drop_depth_ratio']
        ok = ok_b or ok_d
        rules.append({
            'rule_id': 'limited_brush_drop',
            'title': '7点甜区掉拍头与刷球角',
            'status': 'pass' if (ok_b and ok_d) else ('ok' if ok else 'warn'),
            'value': f"{brush_deg or 0:.1f}° / {drop_ratio or 0:.2f}x",
            'unit': 'deg/肩宽',
            'target': f"≥{t['min_brush_angle_deg']:.0f}° 或 ≥{t['min_drop_depth_ratio']:.2f}x",
            'cue': '由下向上掉拍头刷球充分' if ok else '击球前拍头下潜刷球',
        })

    # 3. Contact lateral distance (7-point sweet_spot / ball vs torso center)
    lat_bw = summary.get('contact_lateral_bw')
    if lat_bw is not None:
        ok = lat_bw >= t['min_contact_lateral_bw']
        rules.append({
            'rule_id': 'contact_too_close',
            'title': '触球横向身前空间',
            'status': 'pass' if lat_bw >= 0.75 else ('ok' if ok else 'warn'),
            'value': round(float(lat_bw), 2),
            'unit': '倍肩宽',
            'target': f"≥{t['min_contact_lateral_bw']:.2f}x",
            'cue': '身前横向击球空间充裕' if ok else '击球点离身体远些',
        })

    # 4. 2D Shoulder-to-Spine contraction ratio & dual-view turn span
    rho_delta = summary.get('shoulder_spine_contraction_delta')
    turn_span = summary.get('dual_view_turn_span_deg')
    if rho_delta is not None or turn_span is not None:
        ok_r = rho_delta is not None and rho_delta >= t['min_shoulder_spine_delta']
        ok_t = turn_span is not None and turn_span >= t['min_dual_view_turn_span_deg']
        ok = ok_r or ok_t
        rules.append({
            'rule_id': 'limited_shoulder_turn',
            'title': '肩宽/脊柱收缩比与双视图转肩',
            'status': 'pass' if (ok_r and ok_t) else ('ok' if ok else 'warn'),
            'value': f"Δρ={rho_delta or 0:.2f} / Δθ={turn_span or 0:.1f}°",
            'unit': 'ratio/°',
            'target': f"Δρ≥{t['min_shoulder_spine_delta']:.2f} 或 Δθ≥{t['min_dual_view_turn_span_deg']:.0f}°",
            'cue': '侧身蓄力与转肩释放充分' if ok else '提前侧身转肩充分引拍',
        })

    # 5. 7-point sweet_spot / wrist whip amplification & sequence
    whip = summary.get('whip_amplification_ratio')
    seq_ok = summary.get('wrist_to_racket_sequential')
    if whip is not None:
        ok = whip >= t['min_whip_amplification_ratio'] and (seq_ok is not False)
        rules.append({
            'rule_id': 'insufficient_whip_release',
            'title': '甜区/手腕鞭打加速比',
            'status': 'pass' if (whip >= 2.8 and seq_ok) else ('ok' if ok else 'warn'),
            'value': f"{whip:.2f}x",
            'unit': '倍腕速',
            'target': f"≥{t['min_whip_amplification_ratio']:.1f}x",
            'cue': '手腕制动释放拍头鞭打极佳' if ok else '放松手腕释放拍头鞭打加速',
        })

    warnings = [r for r in rules if r['status'] == 'warn']
    return {
        'policy_version': 'observable_2d_dual_view_v1',
        'approved_rule_count': len(OBSERVABLE_2D_TECHNIQUE_RULES),
        'evaluated_rules': rules,
        'primary_cue': warnings[0]['cue'] if warnings else '本拍2D核心生物力学指标全部达标',
        'overall_status': 'needs_adjustment' if warnings else 'good',
    }

