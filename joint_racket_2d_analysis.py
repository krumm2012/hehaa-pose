"""Source-bound 2D body/racket candidates and missing-observation diagnostics.

No inference, interpolation, 3D conversion or automatic technical scoring.
"""
import math
from collections import Counter
from statistics import median

POLICY = 'joint_racket_2d_v1'
RACKET_POINTS = ('butt_cap', 'grip_center', 'throat', 'sweet_spot', 'head_top', 'frame_left', 'frame_right')
BODY_POINTS = tuple(f'{side}_{joint}' for side in ('left', 'right')
                    for joint in ('shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle'))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def qualified_point(point, frame_id, threshold=.5):
    if not isinstance(point, dict):
        return None, 'missing_point'
    if point.get('source_frame_id') != frame_id or type(point.get('source_frame_id')) is not int:
        return None, 'source_frame_mismatch'
    if point.get('observed') is not True or point.get('recovered_from_mirror'):
        return None, 'derived_or_unobserved'
    if point.get('confidence_source') == 'unavailable':
        return None, 'score_unavailable'
    score = point.get('confidence')
    if not finite(score) or not 0 <= score <= 1:
        return None, 'invalid_score'
    if score < threshold:
        return None, 'low_score'
    if not finite(point.get('x')) or not finite(point.get('y')):
        return None, 'invalid_coordinates'
    return [point['x'], point['y']], None


def source_seconds(row):
    source = row.get('source_time') or {}
    if (source.get('schema_version') == 'tennis.source-time.v1'
            and source.get('source_frame_id') == row['frame_id']
            and source.get('source_kind') == 'video_file'
            and source.get('basis') == 'media_pts' and source.get('quality') == 'reported'
            and finite(source.get('timestamp_seconds')) and source['timestamp_seconds'] >= 0):
        return source['timestamp_seconds']
    return None


def associate_rackets(row, bodies, size, distance_ratio=.05, ambiguity_ratio=.005):
    """Independent wrist/grip proximity candidates, rejecting duplicate view use."""
    fid = row['frame_id']; obs = row.get('racket_pose') or {}
    limit = math.hypot(*size) * distance_ratio
    margin = math.hypot(*size) * ambiguity_ratio
    candidates = obs.get('candidates') or []
    result = {}
    for view in ('front', 'back'):
        wrist_entries = [(side, bodies[view].get(side+'_wrist')) for side in ('left', 'right')]
        wrist_entries = [(s, w) for s, w in wrist_entries if w is not None]
        eligible = []
        reason = 'no_qualified_wrist' if not wrist_entries else 'no_qualified_racket_grip'
        if (wrist_entries and obs.get('frame_id') == fid and obs.get('status') == 'observed'
                and obs.get('coordinate_space') == 'original_source_pixels'):
            for index, candidate in enumerate(candidates):
                if (candidate.get('source') != 'racket_pose_model' or candidate.get('observed') is not True
                        or candidate.get('source_frame_id') != fid
                        or not finite(candidate.get('confidence')) or not .25 <= candidate['confidence'] <= 1):
                    continue
                grip, _ = qualified_point((candidate.get('keypoints') or {}).get('grip_center'), fid)
                if grip is None:
                    continue
                best_dist = float('inf')
                best_alignment = 0.0
                tip, _ = qualified_point((candidate.get('keypoints') or {}).get('throat'), fid)
                if tip is None:
                    tip, _ = qualified_point((candidate.get('keypoints') or {}).get('sweet_spot'), fid)
                for side, w in wrist_entries:
                    d = math.dist(w, grip)
                    if d < best_dist:
                        best_dist = d
                        elbow = bodies[view].get(side+'_elbow')
                        if elbow and tip:
                            arm_v = (w[0]-elbow[0], w[1]-elbow[1])
                            rkt_v = (tip[0]-grip[0], tip[1]-grip[1])
                            norm = math.hypot(*arm_v) * math.hypot(*rkt_v)
                            best_alignment = (arm_v[0]*rkt_v[0] + arm_v[1]*rkt_v[1]) / norm if norm > 1e-6 else 0.0
                        else:
                            best_alignment = 0.0
                if best_dist <= limit:
                    eligible.append((best_dist, -best_alignment, index))
        eligible.sort()
        selected_index = None
        selected_distance = None
        if eligible:
            if len(eligible) == 1:
                selected_distance, _, selected_index = eligible[0]
            elif eligible[1][0] - eligible[0][0] >= margin:
                selected_distance, _, selected_index = eligible[0]
            else:
                # Ambiguity tie-breaker using forearm alignment: if top candidate has distinctly
                # better anatomical collinearity with the arm, resolve the tie.
                align_diff = (-eligible[0][1]) - (-eligible[1][1])
                if align_diff >= 0.25 and (-eligible[0][1]) > 0:
                    selected_distance, _, selected_index = eligible[0]
        if selected_index is not None:
            result[view] = {'status':'proximity_candidate', 'candidate_index':selected_index,
                            'grip_wrist_distance_px':selected_distance, 'reason':None,
                            'person_assignment_verified':False}
        else:
            result[view] = {'status':'unavailable','candidate_index':None,
                            'reason':'ambiguous_grip_assignment' if eligible else reason}
    if (result['front']['candidate_index'] is not None
            and result['front']['candidate_index'] == result['back']['candidate_index']):
        for view in result:
            result[view] = {'status':'unavailable','candidate_index':None,
                            'reason':'same_candidate_competes_for_both_views'}
    return result


def angle(a, b, c):
    if any(p is None for p in (a, b, c)):
        return None
    u = (a[0]-b[0], a[1]-b[1]);v = (c[0]-b[0], c[1]-b[1])
    norm = math.hypot(*u)*math.hypot(*v)
    if norm <= 1e-9:
        return None
    return math.degrees(math.acos(max(-1, min(1, (u[0]*v[0]+u[1]*v[1])/norm))))


def line_angle(a, b):
    if a is None or b is None or math.dist(a,b) < 12:
        return None
    return math.degrees(math.atan2(b[1]-a[1], b[0]-a[0]))


def merge_dual_view_frame(row, bodies, rackets, assoc, candidates, image_size, max_affine_rmse=18.0):
    """Bidirectional dual-view merging for body joints and 7-point racket topology.

    Preserves strict single-view tracks untouched while producing a merged view:
    1. Solves bidirectional torso planar-affine transforms (back->front & front->back).
    2. Heals missing front/back body joints across views when torso alignment passes gate.
    3. Rescues unassociated 7-point racket candidates using cross-view healed wrists.
    4. Projects 7-point racket keypoints across views when one view is occluded/missing.
    """
    from mirror_geometry import estimate_cross_view_torso_transform, map_mirror_point_planar
    fid = row['frame_id']
    obs = row.get('racket_pose') or {}
    valid_obs = (obs.get('frame_id') == fid and obs.get('status') == 'observed'
                 and obs.get('coordinate_space') == 'original_source_pixels')
    diag_len = math.hypot(*image_size)
    rescue_limit = diag_len * 0.065

    t_b2f = estimate_cross_view_torso_transform(
        bodies['back'], bodies['front'], direction='back_to_front', min_conf=0.20, allow_fallback=True
    )
    t_f2b = estimate_cross_view_torso_transform(
        bodies['front'], bodies['back'], direction='front_to_back', min_conf=0.20, allow_fallback=True
    )
    b2f_ok = t_b2f is not None and t_b2f.rmse <= max_affine_rmse
    f2b_ok = t_f2b is not None and t_f2b.rmse <= max_affine_rmse

    merged_body = {'front': {}, 'back': {}}
    body_provenance = {'front': {}, 'back': {}}
    joint_reproj_errors = {}

    for name in BODY_POINTS:
        f_pt = bodies['front'].get(name)
        b_pt = bodies['back'].get(name)
        # Front merged
        if f_pt is not None:
            merged_body['front'][name] = f_pt
            body_provenance['front'][name] = 'observed_front'
        elif b_pt is not None and b2f_ok:
            proj = map_mirror_point_planar(b_pt, affine_result=t_b2f)
            if proj is not None:
                merged_body['front'][name] = [round(proj[0], 2), round(proj[1], 2)]
                body_provenance['front'][name] = 'back_projected_to_front'
            else:
                merged_body['front'][name] = None
                body_provenance['front'][name] = 'missing'
        else:
            merged_body['front'][name] = None
            body_provenance['front'][name] = 'missing'

        # Back merged
        if b_pt is not None:
            merged_body['back'][name] = b_pt
            body_provenance['back'][name] = 'observed_back'
        elif f_pt is not None and f2b_ok:
            proj_b = map_mirror_point_planar(f_pt, affine_result=t_f2b)
            if proj_b is not None:
                merged_body['back'][name] = [round(proj_b[0], 2), round(proj_b[1], 2)]
                body_provenance['back'][name] = 'front_projected_to_back'
            else:
                merged_body['back'][name] = None
                body_provenance['back'][name] = 'missing'
        else:
            merged_body['back'][name] = None
            body_provenance['back'][name] = 'missing'

        if f_pt is not None and b_pt is not None and b2f_ok:
            proj = map_mirror_point_planar(b_pt, affine_result=t_b2f)
            if proj is not None:
                joint_reproj_errors[name] = round(math.dist(proj, f_pt), 2)

    # Step 3: Rescue unassociated native 7-point racket candidates using merged wrists
    used_indices = {assoc[v]['candidate_index'] for v in ('front', 'back') if assoc[v]['candidate_index'] is not None}
    merged_racket = {'front': dict(rackets['front']), 'back': dict(rackets['back'])}
    racket_provenance = {
        'front': 'observed_front' if assoc['front']['candidate_index'] is not None else 'missing',
        'back': 'observed_back' if assoc['back']['candidate_index'] is not None else 'missing',
    }
    rescued_candidate_index = {'front': None, 'back': None}

    if valid_obs and candidates:
        for view in ('front', 'back'):
            if assoc[view]['candidate_index'] is not None:
                continue
            wrists = [(s, merged_body[view].get(s + '_wrist')) for s in ('left', 'right')]
            wrists = [(s, w) for s, w in wrists if w is not None]
            if not wrists:
                continue
            best_cand = None
            for idx, cand in enumerate(candidates):
                if idx in used_indices:
                    continue
                if (cand.get('source') != 'racket_pose_model' or cand.get('observed') is not True
                        or cand.get('source_frame_id') != fid
                        or not finite(cand.get('confidence')) or not 0.25 <= cand['confidence'] <= 1):
                    continue
                kpts = cand.get('keypoints') or {}
                grip, _ = qualified_point(kpts.get('grip_center'), fid, threshold=0.35)
                if grip is None:
                    grip, _ = qualified_point(kpts.get('butt_cap'), fid, threshold=0.35)
                if grip is None:
                    continue
                for side, w in wrists:
                    d = math.dist(w, grip)
                    if d <= rescue_limit and (best_cand is None or d < best_cand[0]):
                        best_cand = (d, idx, cand)
            if best_cand is not None:
                _, r_idx, r_cand = best_cand
                kpts = r_cand.get('keypoints') or {}
                extracted = {}
                for name in RACKET_POINTS:
                    xy, _ = qualified_point(kpts.get(name), fid, threshold=0.35)
                    extracted[name] = xy
                if any(v is not None for v in extracted.values()):
                    merged_racket[view] = extracted
                    used_indices.add(r_idx)
                    rescued_candidate_index[view] = r_idx
                    racket_provenance[view] = (
                        'healed_wrist_candidate_rescue'
                        if body_provenance[view].get('right_wrist') != f'observed_{view}'
                        or body_provenance[view].get('left_wrist') != f'observed_{view}'
                        else 'relaxed_threshold_candidate_rescue'
                    )

    # Step 4: Cross-view 7-point racket projection when one view's racket is still missing.
    # Anchor relative racket vector (p - grip_center) to the destination wrist when available
    # so 3D torso-to-hand depth parallax under a 35-deg tilted camera is cancelled out.
    def _project_racket_cross_view(src_view, dst_view, tf, prov_tag):
        if tf is None:
            return
        src_rkt = merged_racket[src_view]
        dst_rkt = merged_racket[dst_view]
        src_grip = src_rkt.get('grip_center') or src_rkt.get('butt_cap')
        dst_wrist = merged_body[dst_view].get('right_wrist') or merged_body[dst_view].get('left_wrist')
        use_wrist_anchor = src_grip is not None and dst_wrist is not None
        sx, sy = tf.scale_x, tf.scale_y
        for name in RACKET_POINTS:
            if dst_rkt.get(name) is None and src_rkt.get(name) is not None:
                sp = src_rkt[name]
                if use_wrist_anchor:
                    px = dst_wrist[0] + sx * (sp[0] - src_grip[0])
                    py = dst_wrist[1] + sy * (sp[1] - src_grip[1])
                    dst_rkt[name] = [round(px, 2), round(py, 2)]
                else:
                    proj = map_mirror_point_planar(sp, affine_result=tf)
                    if proj is not None:
                        dst_rkt[name] = [round(proj[0], 2), round(proj[1], 2)]
                if dst_rkt.get(name) is not None and racket_provenance[dst_view] == 'missing':
                    racket_provenance[dst_view] = prov_tag

    if b2f_ok:
        _project_racket_cross_view('back', 'front', t_b2f, 'back_projected_to_front')
    if f2b_ok:
        _project_racket_cross_view('front', 'back', t_f2b, 'front_projected_to_back')

    racket_reproj_error = None
    if (rackets['front'].get('sweet_spot') is not None
            and rackets['back'].get('sweet_spot') is not None and b2f_ok):
        proj_ss = map_mirror_point_planar(rackets['back']['sweet_spot'], affine_result=t_b2f)
        if proj_ss is not None:
            racket_reproj_error = round(math.dist(proj_ss, rackets['front']['sweet_spot']), 2)

    return {
        'torso_transform_status': t_b2f.status if t_b2f is not None else 'unavailable',
        'torso_affine_rmse_px': t_b2f.rmse if t_b2f is not None else None,
        'scale_x': round(t_b2f.scale_x, 4) if t_b2f is not None else None,
        'scale_y': round(t_b2f.scale_y, 4) if t_b2f is not None else None,
        'merged_body': merged_body,
        'body_provenance': body_provenance,
        'merged_racket': merged_racket,
        'racket_provenance': racket_provenance,
        'rescued_candidate_index': rescued_candidate_index,
        'joint_reprojection_errors_px': joint_reproj_errors,
        'sweet_spot_reprojection_error_px': racket_reproj_error,
    }


def summarize_dual_view_fusion(frames):
    total = len(frames)
    if not total:
        return {}
    stats = {}
    for view in ('front', 'back'):
        for joint in ('right_wrist', 'left_wrist', 'right_elbow', 'left_elbow'):
            raw_n = sum(1 for f in frames if f['body'][view].get(joint) is not None)
            merged_n = sum(1 for f in frames if (f.get('dual_view_fusion') or {}).get('merged_body', {}).get(view, {}).get(joint) is not None)
            stats[f'{view}.body.{joint}'] = {
                'raw_frames': raw_n,
                'merged_frames': merged_n,
                'gained_frames': merged_n - raw_n,
                'raw_coverage': round(raw_n / total, 4),
                'merged_coverage': round(merged_n / total, 4),
            }
        raw_rkt = sum(1 for f in frames if f['racket'][view].get('sweet_spot') is not None)
        merged_rkt = sum(1 for f in frames if (f.get('dual_view_fusion') or {}).get('merged_racket', {}).get(view, {}).get('sweet_spot') is not None)
        rescued_rkt = sum(1 for f in frames if (f.get('dual_view_fusion') or {}).get('rescued_candidate_index', {}).get(view) is not None)
        projected_rkt = sum(
            1 for f in frames
            if (f.get('dual_view_fusion') or {}).get('racket_provenance', {}).get(view) in ('back_projected_to_front', 'front_projected_to_back')
        )
        stats[f'{view}.racket.sweet_spot'] = {
            'raw_frames': raw_rkt,
            'merged_frames': merged_rkt,
            'gained_frames': merged_rkt - raw_rkt,
            'rescued_native_candidate_frames': rescued_rkt,
            'cross_view_projected_frames': projected_rkt,
            'raw_coverage': round(raw_rkt / total, 4),
            'merged_coverage': round(merged_rkt / total, 4),
        }
    return stats


def analyze_frames(rows, image_size):
    from ball_observation_contract import measurement_ball
    ordered = sorted(rows, key=lambda r:r['frame_id'])
    ids = [r['frame_id'] for r in ordered]
    if any(type(fid) is not int or fid < 0 for fid in ids) or len(set(ids)) != len(ids):
        raise ValueError('Source frame identities must be unique nonnegative integers')
    times = [source_seconds(r) for r in ordered]
    dts = [b-a for a,b in zip(times,times[1:]) if a is not None and b is not None and b>a]
    cadence = median(dts) if dts else None
    frames = []
    for row, timestamp in zip(ordered,times):
        fid = row['frame_id'];bodies={};reasons={};tracks={}
        if row.get('pose_observation_coordinate_space') != 'original_source_pixels':
            observations = {}
        else:
            observations = row.get('pose_observations') or {}
        for view in ('front','back'):
            bodies[view] = {}
            for name in BODY_POINTS:
                xy, reason = qualified_point((observations.get(view)or{}).get(name),fid)
                bodies[view][name]=xy;key=f'{view}.body.{name}'
                tracks[key]=xy;reasons[key]=reason
        assoc=associate_rackets(row,bodies,image_size)
        obs=row.get('racket_pose')or{}; candidates=obs.get('candidates')or[]
        metrics={};rackets={}
        for view in ('front','back'):
            index=assoc[view]['candidate_index'];rackets[view]={}
            candidate=candidates[index] if index is not None else {}
            for name in RACKET_POINTS:
                xy, reason = qualified_point((candidate.get('keypoints')or{}).get(name),fid)
                if index is None:reason=assoc[view]['reason']
                rackets[view][name]=xy;key=f'{view}.racket.{name}'
                tracks[key]=xy;reasons[key]=reason
            body=bodies[view];rkt=rackets[view]
            width=math.dist(body['left_shoulder'],body['right_shoulder']) if body['left_shoulder'] and body['right_shoulder'] else None
            if width is not None and width<12:width=None
            sh_mid=[(body['left_shoulder'][0]+body['right_shoulder'][0])/2,(body['left_shoulder'][1]+body['right_shoulder'][1])/2] if body['left_shoulder'] and body['right_shoulder'] else (body['left_shoulder'] or body['right_shoulder'])
            hip_mid=[(body['left_hip'][0]+body['right_hip'][0])/2,(body['left_hip'][1]+body['right_hip'][1])/2] if body['left_hip'] and body['right_hip'] else (body['left_hip'] or body['right_hip'])
            spine_len=math.dist(sh_mid,hip_mid) if sh_mid and hip_mid else None
            if spine_len is not None and spine_len<20:spine_len=None
            sh_spine_ratio=round(width/spine_len,3) if width is not None and spine_len is not None else None
            hip=line_angle(body['left_hip'],body['right_hip']);shoulder=line_angle(body['left_shoulder'],body['right_shoulder'])
            metrics[view]={'body_width_px':width,'spine_length_px':spine_len,'shoulder_spine_ratio':sh_spine_ratio,
                'hip_line_deg':hip,'shoulder_line_deg':shoulder,
                'hip_shoulder_projection_deg':abs((shoulder-hip+90)%180-90) if hip is not None and shoulder is not None else None,
                'shaft_image_angle_deg':line_angle(rkt['butt_cap'],rkt['head_top']),
                'grip_wrist_gap_body_widths':assoc[view].get('grip_wrist_distance_px')/width if width and index is not None else None}
            for side in ('left','right'):
                metrics[view][side+'_elbow_deg']=angle(body[side+'_shoulder'],body[side+'_elbow'],body[side+'_wrist'])
                metrics[view][side+'_knee_deg']=angle(body[side+'_hip'],body[side+'_knee'],body[side+'_ankle'])
        for view in ('front','back'):
            for group,names in (('arm_chain',('right_shoulder','right_elbow','right_wrist')),('left_arm_chain',('left_shoulder','left_elbow','left_wrist')),('torso',('left_hip','right_hip','left_shoulder','right_shoulder'))):
                key=f'{view}.complete.{group}';tracks[key]=[0,0] if all(bodies[view][name] is not None for name in names) else None;reasons[key]=None if tracks[key] is not None else 'incomplete_body_chain'
            key=f'{view}.complete.seven_points';tracks[key]=[0,0] if all(rackets[view][name] is not None for name in RACKET_POINTS) else None;reasons[key]=None if tracks[key] is not None else 'incomplete_associated_racket'
        ball, origin=measurement_ball(row)
        if origin != 'fresh_model_observation':ball=None
        tracks['ball']=ball;reasons['ball']=None if ball else 'no_qualified_active_ball'
        tracks['racket_detector_any']=[0,0] if obs.get('frame_id')==fid and obs.get('status')=='observed' and candidates else None
        reasons['racket_detector_any']=None if tracks['racket_detector_any'] else 'no_raw_racket_candidate'
        for view in ('front','back'):
            metrics[view]['ball_sweetspot_distance_px']=math.dist(ball,rackets[view]['sweet_spot']) if view=='front' and ball is not None and rackets[view]['sweet_spot'] is not None else None
        fusion = merge_dual_view_frame(row, bodies, rackets, assoc, candidates, image_size)
        fw = metrics['front']['body_width_px'];bw = metrics['back']['body_width_px']
        sx = fusion.get('scale_x') or 1.48
        dv_turn = round(math.degrees(math.atan2(bw * sx, fw)), 1) if (fw and bw and fw >= 14 and bw >= 12) else None
        fusion['dual_view_turn_deg'] = dv_turn
        metrics['front']['dual_view_turn_deg'] = dv_turn
        metrics['back']['dual_view_turn_deg'] = dv_turn
        frame={'frame_id':fid,'source_seconds':timestamp,'body':bodies,'racket':rackets,'associations':assoc,
               'ball':ball,'metrics':metrics,'tracks':tracks,'missing_reasons':reasons,'dual_view_fusion':fusion}
        previous=frames[-1] if frames else None
        for view in ('front','back'):
            m=metrics[view];m.update(sweet_spot_speed_px_s=None,wrist_speed_px_s=None)
            dt=timestamp-previous['source_seconds'] if previous and timestamp is not None and previous['source_seconds'] is not None else None
            valid_dt=(dt is not None and cadence is not None and .25*cadence<dt<=3*cadence
                      and fid==previous['frame_id']+1)
            m['velocity_status']='qualified_source_interval' if valid_dt else 'missing_or_unstable_source_interval'
            for metric,point in (('sweet_spot_speed_px_s',view+'.racket.sweet_spot'),('wrist_speed_px_s',view+'.body.right_wrist')):
                a=previous['tracks'][point] if previous else None;b=tracks[point]
                if valid_dt and a is not None and b is not None:m[metric]=math.dist(a,b)/dt
        frames.append(frame)
    _compute_merged_kinematics(frames, cadence)
    return frames,cadence


def _compute_merged_kinematics(frames, cadence):
    """Compute spike-filtered, short-gap-bridged, central-difference speed curves on merged tracks."""
    n = len(frames)
    if not n or cadence is None:
        for f in frames:
            if 'dual_view_fusion' in f:
                f['dual_view_fusion']['kinematics'] = {
                    v: {'wrist_speed_px_s': None, 'sweet_spot_speed_px_s': None} for v in ('front', 'back')
                }
        return

    def _valid_step(i_prev, i_curr):
        f0, f1 = frames[i_prev], frames[i_curr]
        if f1['frame_id'] - f0['frame_id'] != i_curr - i_prev:
            return False, None
        t0, t1 = f0['source_seconds'], f1['source_seconds']
        if t0 is None or t1 is None or t1 <= t0:
            return False, None
        dt = t1 - t0
        span = i_curr - i_prev
        if not (0.25 * cadence * span < dt <= 3.0 * cadence * span):
            return False, None
        return True, dt

    for view in ('front', 'back'):
        scale_factor = 1.0 if view == 'front' else 0.68

        for kind, pt_name, out_key in (
            ('merged_body', 'right_wrist', 'wrist_speed_px_s'),
            ('merged_racket', 'sweet_spot', 'sweet_spot_speed_px_s'),
        ):
            is_racket = pt_name == 'sweet_spot'
            max_jump_px = (240.0 if is_racket else 125.0) * scale_factor
            spike_trigger_px = (75.0 if is_racket else 42.0) * scale_factor
            max_bridge_gap = 3 if is_racket else 2

            pts = []
            for f in frames:
                dv = f.get('dual_view_fusion') or {}
                p = (dv.get(kind) or {}).get(view, {}).get(pt_name)
                pts.append(list(p) if p is not None else None)

            # 1. Suppress 1-frame A->B->A teleportation spikes (including pre/post short-gap spikes)
            for i in range(1, n - 1):
                if pts[i] is None:
                    continue
                i_prev = next((p for p in range(i - 1, max(-1, i - 2 - max_bridge_gap), -1) if pts[p] is not None), None)
                i_next = next((q for q in range(i + 1, min(n, i + 2 + max_bridge_gap)) if pts[q] is not None), None)
                if i_prev is None or i_next is None or (i_next - i_prev) > (2 + max_bridge_gap):
                    continue
                ok1, dt1 = _valid_step(i_prev, i)
                ok2, dt2 = _valid_step(i, i_next)
                if not (ok1 and ok2):
                    continue
                d_in = math.dist(pts[i_prev], pts[i])
                d_out = math.dist(pts[i], pts[i_next])
                d_span = math.dist(pts[i_prev], pts[i_next])
                if max(d_in, d_out) > spike_trigger_px and (d_in + d_out) > 2.1 * max(10.0, d_span):
                    alpha = dt1 / (dt1 + dt2)
                    pts[i] = [
                        round(pts[i_prev][0] + alpha * (pts[i_next][0] - pts[i_prev][0]), 2),
                        round(pts[i_prev][1] + alpha * (pts[i_next][1] - pts[i_prev][1]), 2),
                    ]
                    dv_i = frames[i].get('dual_view_fusion') or {}
                    if pt_name == 'right_wrist':
                        dv_i['merged_body'][view][pt_name] = pts[i]
                        dv_i['body_provenance'][view][pt_name] = 'spike_suppressed_interp'
                    else:
                        dv_i['merged_racket'][view][pt_name] = pts[i]
                        dv_i['racket_provenance'][view] = 'spike_suppressed_interp'

            # 2. Bridge short motion-blur gaps (1-3 frames for racket sweet_spot, 1-2 for wrist)
            idx = 0
            while idx < n:
                if pts[idx] is not None:
                    idx += 1
                    continue
                g_start = idx
                while idx < n and pts[idx] is None:
                    idx += 1
                g_end = idx
                g_len = g_end - g_start
                if 1 <= g_len <= max_bridge_gap and g_start > 0 and g_end < n:
                    ok_span, dt_span = _valid_step(g_start - 1, g_end)
                    if ok_span and dt_span is not None:
                        p_a, p_b = pts[g_start - 1], pts[g_end]
                        dist_ab = math.dist(p_a, p_b)
                        if dist_ab <= max_jump_px * (g_len + 1):
                            t_a = frames[g_start - 1]['source_seconds']
                            # Check if internal timestamps are monotonic and well-spaced; otherwise use uniform index alpha
                            internal_ok = all(_valid_step(m - 1, m)[0] and _valid_step(m - 1, m)[1] >= 0.4 * cadence for m in range(g_start, g_end + 1))
                            for k in range(g_start, g_end):
                                tk = frames[k]['source_seconds']
                                alpha = (tk - t_a) / dt_span if (internal_ok and tk is not None) else (k - g_start + 1) / (g_len + 1)
                                pts[k] = [
                                    round(p_a[0] + alpha * (p_b[0] - p_a[0]), 2),
                                    round(p_a[1] + alpha * (p_b[1] - p_a[1]), 2),
                                ]
                                dv_k = frames[k].get('dual_view_fusion') or {}
                                if pt_name == 'right_wrist':
                                    dv_k['merged_body'][view][pt_name] = pts[k]
                                    dv_k['body_provenance'][view][pt_name] = 'short_gap_bridged'
                                else:
                                    dv_k['merged_racket'][view][pt_name] = pts[k]
                                    if dv_k['racket_provenance'].get(view) == 'missing':
                                        dv_k['racket_provenance'][view] = 'short_gap_bridged'

            # 3. Compute PTS-weighted central difference speed (eliminates 30ms/48ms PTS jitter sawtooth)
            raw_speeds = [None] * n
            min_step_dt = 0.80 * cadence
            for i in range(n):
                if pts[i] is None:
                    continue
                has_prev = i > 0 and pts[i - 1] is not None and _valid_step(i - 1, i)[0]
                has_next = i + 1 < n and pts[i + 1] is not None and _valid_step(i, i + 1)[0]
                if has_prev and has_next:
                    _, dt1 = _valid_step(i - 1, i)
                    _, dt2 = _valid_step(i, i + 1)
                    d1 = math.dist(pts[i - 1], pts[i])
                    d2 = math.dist(pts[i], pts[i + 1])
                    if d1 <= max_jump_px * 1.3 and d2 <= max_jump_px * 1.3:
                        raw_speeds[i] = (d1 + d2) / max(dt1 + dt2, 2.0 * min_step_dt)
                elif has_prev:
                    _, dt1 = _valid_step(i - 1, i)
                    d1 = math.dist(pts[i - 1], pts[i])
                    if d1 <= max_jump_px * 1.2:
                        raw_speeds[i] = d1 / max(dt1, min_step_dt)
                elif has_next:
                    _, dt2 = _valid_step(i, i + 1)
                    d2 = math.dist(pts[i], pts[i + 1])
                    if d2 <= max_jump_px * 1.2:
                        raw_speeds[i] = d2 / max(dt2, min_step_dt)

            # 4. 3-point [0.25, 0.50, 0.25] zero-phase smoothing over consecutive valid frames
            smoothed_speeds = [None] * n
            for i in range(n):
                if raw_speeds[i] is None:
                    continue
                s_prev = raw_speeds[i - 1] if (i > 0 and raw_speeds[i - 1] is not None and _valid_step(i - 1, i)[0]) else None
                s_next = raw_speeds[i + 1] if (i + 1 < n and raw_speeds[i + 1] is not None and _valid_step(i, i + 1)[0]) else None
                if s_prev is not None and s_next is not None:
                    smoothed_speeds[i] = round(0.25 * s_prev + 0.50 * raw_speeds[i] + 0.25 * s_next, 1)
                elif s_prev is not None:
                    smoothed_speeds[i] = round(0.35 * s_prev + 0.65 * raw_speeds[i], 1)
                elif s_next is not None:
                    smoothed_speeds[i] = round(0.65 * raw_speeds[i] + 0.35 * s_next, 1)
                else:
                    smoothed_speeds[i] = round(raw_speeds[i], 1)

            for i, f in enumerate(frames):
                dv = f.setdefault('dual_view_fusion', {})
                kin = dv.setdefault('kinematics', {'front': {}, 'back': {}})
                kin[view][out_key] = smoothed_speeds[i]
                kin[view]['raw_' + out_key] = round(raw_speeds[i], 1) if raw_speeds[i] is not None else None


def missing_runs(frames, key, minimum_length=3):
    """Observation gaps, split at absent journal IDs; never imply decoded loss."""
    runs=[];current=[]
    def close():
        if len(current)>=minimum_length:
            first,last=current[0],current[-1]
            runs.append({'track':key,'start_frame':first['frame_id'],'end_frame':last['frame_id'],
                         'count':len(current),'start_seconds':first['source_seconds'],'end_seconds':last['source_seconds'],
                         'pts_span_ms':(last['source_seconds']-first['source_seconds'])*1000
                         if first['source_seconds'] is not None and last['source_seconds'] is not None else None,
                         'reasons':dict(Counter(f['missing_reasons'].get(key) for f in current))})
    for f in frames:
        if current and f['frame_id']!=current[-1]['frame_id']+1:close();current=[]
        if f['tracks'].get(key) is None:current.append(f)
        else:close();current=[]
    close();return runs


def summarize(frames, events, expected_source_frame_count=None):
    keys=list(frames[0]['tracks']) if frames else []
    summary=[];runs=[]
    for key in keys:
        all_gaps=missing_runs(frames,key,1)
        gaps=[gap for gap in all_gaps if gap['count']>=3]
        for gap in gaps:
            gap['event_ids']=[e['event_id'] for e in events if gap['end_frame']>=e['start_frame'] and gap['start_frame']<=e['end_frame']]
            contact=[]
            for e in events:
                row=next((f for f in frames if f['frame_id']==e['contact_frame']),None)
                t=row['source_seconds'] if row else None
                if t is not None and gap['start_seconds'] is not None and gap['end_seconds'] is not None:
                    if gap['end_seconds']>=t-.12 and gap['start_seconds']<=t+.12:contact.append(e['event_id'])
            gap['near_contact_event_ids']=contact
            gap['priority']='contact_window' if contact else 'swing_window' if gap['event_ids'] else 'outside_event'
        runs.extend(gaps)
        n=sum(f['tracks'].get(key) is not None for f in frames)
        summary.append({'track':key,'qualified_frames':n,'total_frames':len(frames),
                        'coverage':n/len(frames) if frames else 0,'runs_ge_3':len(gaps),
                        'exactly_3':sum(g['count']==3 for g in gaps),'runs_1':sum(g['count']==1 for g in all_gaps),'runs_2':sum(g['count']==2 for g in all_gaps),'longest_run':max((g['count'] for g in all_gaps),default=0)})
    journal_gaps=[{'start_frame':a['frame_id']+1,'end_frame':b['frame_id']-1,'count':b['frame_id']-a['frame_id']-1}
                  for a,b in zip(frames,frames[1:]) if b['frame_id']>a['frame_id']+1]
    if expected_source_frame_count is not None:
        if type(expected_source_frame_count) is not int or expected_source_frame_count<0 or any(f['frame_id']>=expected_source_frame_count for f in frames):
            raise ValueError('Invalid decoded source frame count')
        missing=sorted(set(range(expected_source_frame_count))-{f['frame_id'] for f in frames})
        journal_gaps=[]
        for fid in missing:
            if journal_gaps and fid==journal_gaps[-1]['end_frame']+1:
                journal_gaps[-1]['end_frame']=fid;journal_gaps[-1]['count']+=1
            else:journal_gaps.append({'start_frame':fid,'end_frame':fid,'count':1})
    return {'tracks':summary,'observation_gaps_ge_3':runs,'journal_gaps':journal_gaps,'expected_source_frame_count':expected_source_frame_count,
            'dual_view_fusion':summarize_dual_view_fusion(frames),
            'semantics':'Point unavailable includes missing, low score, unverified assignment; not visibility truth or detector recall. PTS span is last-first, not exposure duration.',
            'accuracy_validated':False,'coach_eligible':False}


def summarize_event_2d_diagnostics(frames, event):
    """Compute 2D biomechanical metrics (7-pt brush angle, lateral distance, shoulder-spine contraction, whip) and evaluate 2D rules."""
    from coach_rule_contract import evaluate_2d_observable_diagnostics
    start_f = int(event['start_frame'])
    end_f = int(event['end_frame'])
    contact_f = int(event.get('contact_frame', start_f))
    ev_frames = [f for f in frames if start_f <= f['frame_id'] <= end_f]
    by_id = {f['frame_id']: f for f in frames}
    cf = by_id.get(contact_f) or (ev_frames[-1] if ev_frames else None)
    if not cf or not ev_frames:
        return {}

    # Robust reference scale from spine length or shoulder width
    scales = []
    for f in ev_frames:
        m = f['metrics']['front']
        bw = m.get('body_width_px')
        sl = m.get('spine_length_px')
        if sl and sl > 25:
            scales.append(max(bw or 0.0, sl / 1.40))
        elif bw and bw > 20:
            scales.append(bw)
    ref_scale = sorted(scales)[len(scales) // 2] if scales else 100.0

    def _best_racket_pt(row):
        mr = (row.get('dual_view_fusion') or {}).get('merged_racket', {}).get('front', {})
        return mr.get('sweet_spot') or mr.get('head_top') or row['racket']['front'].get('sweet_spot')

    # Find racket drop lowest point (max Y) prior to follow-through, then find forward-swing sweet_spot peak after it
    drop_candidates = [
        (f['frame_id'], _best_racket_pt(f))
        for f in ev_frames
        if _best_racket_pt(f) is not None and f['frame_id'] <= contact_f + 2
    ]
    lowest_drop_f = max(drop_candidates, key=lambda x: x[1][1])[0] if drop_candidates else start_f

    all_s_speeds = [
        (f['frame_id'], (f.get('dual_view_fusion') or {}).get('kinematics', {}).get('front', {}).get('sweet_spot_speed_px_s'))
        for f in ev_frames
        if f['frame_id'] >= lowest_drop_f and abs(f['frame_id'] - contact_f) <= 9
    ]
    all_s_speeds = [(fid, v) for fid, v in all_s_speeds if v is not None]
    if not all_s_speeds:
        all_s_speeds = [
            (f['frame_id'], (f.get('dual_view_fusion') or {}).get('kinematics', {}).get('front', {}).get('sweet_spot_speed_px_s'))
            for f in ev_frames
            if (f.get('dual_view_fusion') or {}).get('kinematics', {}).get('front', {}).get('sweet_spot_speed_px_s') is not None
        ]
    s_peak_f, max_s = max(all_s_speeds, key=lambda x: x[1]) if all_s_speeds else (None, None)
    impact_f = s_peak_f if (s_peak_f is not None and abs(contact_f - s_peak_f) <= 10) else contact_f
    cf = by_id.get(impact_f) or cf

    # Forward-swing wrist peak in [impact_f - 5, impact_f + 2]
    whip_w_rows = [f for f in ev_frames if (impact_f - 5) <= f['frame_id'] <= (impact_f + 2)]
    w_speeds = [
        (f['frame_id'], (f.get('dual_view_fusion') or {}).get('kinematics', {}).get('front', {}).get('wrist_speed_px_s'))
        for f in (whip_w_rows or ev_frames)
    ]
    w_speeds = [(fid, v) for fid, v in w_speeds if v is not None]
    w_peak_f, max_w = max(w_speeds, key=lambda x: x[1]) if w_speeds else (None, None)
    whip_ratio = round(max_s / max_w, 2) if (max_s and max_w and max_w > 80.0) else None
    seq_ok = (w_peak_f <= s_peak_f + 1) if (w_peak_f is not None and s_peak_f is not None) else None

    # 1. Contact arm extension (right_elbow_deg at impact_f +/- 1 frame)
    elbow_cands = []
    for fid in (impact_f, impact_f - 1, impact_f + 1):
        row = by_id.get(fid)
        if not row:
            continue
        eb = row['metrics']['front'].get('right_elbow_deg')
        if eb is None:
            mb = (row.get('dual_view_fusion') or {}).get('merged_body', {}).get('front', {})
            eb = angle(mb.get('right_shoulder'), mb.get('right_elbow'), mb.get('right_wrist'))
        if eb is not None:
            elbow_cands.append(eb)
    contact_elbow_deg = round(max(elbow_cands), 1) if elbow_cands else None

    # 2. 7-point sweet_spot / head_top brush angle & drop depth in [-0.48s, 0s] before impact_f
    t_contact = cf.get('source_seconds')
    pre_rows = [
        f for f in frames
        if (t_contact is not None and f.get('source_seconds') is not None and (t_contact - 0.48 - 1e-6) <= f['source_seconds'] <= (t_contact + 0.05 + 1e-6))
        or (t_contact is None and max(start_f, impact_f - 12) <= f['frame_id'] <= impact_f + 1)
    ]
    rkt_pts = [(f['frame_id'], _best_racket_pt(f)) for f in pre_rows if _best_racket_pt(f) is not None and f['frame_id'] <= impact_f]
    c_pt = _best_racket_pt(cf)
    if c_pt is None and impact_f + 1 in by_id:
        c_pt = _best_racket_pt(by_id[impact_f + 1])
    # For upward follow-through trajectory angle, also check impact_f + 1 if impact_f is near bottom of arc
    end_pt = _best_racket_pt(by_id[impact_f + 1]) if (impact_f + 1 in by_id and _best_racket_pt(by_id[impact_f + 1]) is not None) else c_pt
    brush_angle_deg = None
    drop_depth_px = None
    drop_depth_ratio = None
    lowest_frame = None
    if len(rkt_pts) >= 2 and end_pt is not None:
        lowest_frame, low_pt = max(rkt_pts, key=lambda x: x[1][1])
        dy = max(0.0, low_pt[1] - end_pt[1])
        dx = abs(end_pt[0] - low_pt[0])
        if dy > 0 or dx > 0:
            brush_angle_deg = round(math.degrees(math.atan2(dy, dx + 1e-5)), 1)
            drop_depth_px = round(dy, 1)
            drop_depth_ratio = round(dy / ref_scale, 2)

    # 3. Contact lateral distance (7-point sweet_spot vs torso center X, in spine-calibrated body widths)
    mb_c = (cf.get('dual_view_fusion') or {}).get('merged_body', {}).get('front', {})
    sh_pts = [mb_c.get('left_shoulder'), mb_c.get('right_shoulder'), mb_c.get('left_hip'), mb_c.get('right_hip')]
    sh_pts = [p for p in sh_pts if p is not None]
    torso_cx = sum(p[0] for p in sh_pts) / len(sh_pts) if sh_pts else None
    target_contact_xy = c_pt or mb_c.get('right_wrist')
    contact_lateral_bw = (
        round(abs(target_contact_xy[0] - torso_cx) / ref_scale, 2)
        if target_contact_xy is not None and torso_cx is not None and ref_scale > 0
        else None
    )

    # 4. 2D Shoulder-to-Spine contraction ratio & Dual-View turn span
    prep_rows = [f for f in ev_frames if f['frame_id'] <= impact_f + 2]
    rhos = [f['metrics']['front']['shoulder_spine_ratio'] for f in prep_rows if f['metrics']['front'].get('shoulder_spine_ratio') is not None]
    dv_turns = [f['metrics']['front']['dual_view_turn_deg'] for f in prep_rows if f['metrics']['front'].get('dual_view_turn_deg') is not None]
    rho_min = round(min(rhos), 3) if rhos else None
    rho_max = round(max(rhos), 3) if rhos else None
    rho_delta = round(rho_max - rho_min, 3) if (rho_min is not None and rho_max is not None) else None
    turn_span = round(max(dv_turns) - min(dv_turns), 1) if len(dv_turns) >= 2 else None

    summary = {
        'reference_scale_px': round(ref_scale, 1),
        'kinematic_impact_frame': impact_f,
        'contact_elbow_deg': contact_elbow_deg,
        'brush_angle_deg': brush_angle_deg,
        'drop_depth_px': drop_depth_px,
        'drop_depth_ratio': drop_depth_ratio,
        'lowest_racket_frame': lowest_frame,
        'contact_lateral_bw': contact_lateral_bw,
        'shoulder_spine_min': rho_min,
        'shoulder_spine_max': rho_max,
        'shoulder_spine_contraction_delta': rho_delta,
        'dual_view_turn_span_deg': turn_span,
        'max_wrist_speed_px_s': max_w,
        'max_sweet_spot_speed_px_s': max_s,
        'wrist_peak_frame': w_peak_f,
        'sweet_spot_peak_frame': s_peak_f,
        'whip_amplification_ratio': whip_ratio,
        'wrist_to_racket_sequential': seq_ok,
    }
    summary['coach_evaluation'] = evaluate_2d_observable_diagnostics(summary)
    return summary
