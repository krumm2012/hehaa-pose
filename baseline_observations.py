"""Causal per-view baseline diagnostics, never calibrated uncertainty or 3D truth."""
import math
from statistics import median
from image_motion_measurements import source_timestamp

VERSION = 'tennis.baseline-observation.v1'
PARAMETERS = {'window_seconds_before_start': [-.8, -.12], 'min_samples': 6,
              'min_duration_seconds': .2, 'min_coverage': .8,
              'max_scale_range_ratio': .1, 'max_hip_drift_torso_ratio': .03,
              'max_shoulder_angle_range_deg': 8}


def _geometry(row, view):
    pose = (row.get('pose_observations') or {}).get(view, {})
    points = []
    for name in ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'):
        p = pose.get(name, {})
        values = [p.get(k) for k in ('x','y','confidence')]
        if (p.get('observed') is not True or p.get('recovered_from_mirror')
                or p.get('confidence_source') == 'unavailable'
                or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in values)
                or values[2] < .5):
            return None
        points.append(values[:2])
    ls,rs,lh,rh = points
    shoulder = [(a+b)/2 for a,b in zip(ls,rs)]
    hip = [(a+b)/2 for a,b in zip(lh,rh)]
    torso, width = math.dist(shoulder,hip), math.dist(ls,rs)
    if torso <= 0 or width <= 0:
        return None
    return {'torso':torso,'width':width,'hip':hip,
            'angle':math.degrees(math.atan2(rs[1]-ls[1],rs[0]-ls[0]))}


def baseline_profiles(frames, start_frame):
    rows = sorted(frames, key=lambda r:r['frame_id'])
    anchor = next((r for r in rows if r['frame_id']==start_frame), {})
    t0,basis = source_timestamp(anchor)
    result = {'version':VERSION,'parameters':PARAMETERS.copy(),'views':{},
              'accuracy_validated':False,'scope':'experimental_baseline_diagnostics_only'}
    for view in ('front','back'):
        reasons=[]; window=[]; samples=[]
        prior=[r for r in rows if r['frame_id']<start_frame]
        if t0 is None or basis not in ('media_pts','nominal_fps'):
            reasons.append('source_time_unavailable')
        else:
            for row in prior:
                t,b=source_timestamp(row)
                if t is None or b!=basis:
                    # Cannot establish whether this unknown sample belongs to baseline.
                    reasons.append('unlocated_source_time');continue
                if t0-.8-1e-9 <= t <= t0-.12+1e-9:
                    window.append((row,t))
            if any(b[1]<=a[1] or b[0]['frame_id']!=a[0]['frame_id']+1 for a,b in zip(window,window[1:])):
                reasons.append('baseline_discontinuous')
            for row,t in window:
                g=_geometry(row,view)
                if g is not None:samples.append((row['frame_id'],t,g))
        coverage=len(samples)/max(1,len(window))
        if len(samples)<6:reasons.append('insufficient_baseline_samples')
        if coverage<.8:reasons.append('low_baseline_coverage')
        if not samples or samples[-1][1]-samples[0][1]<.2:reasons.append('insufficient_baseline_duration')
        diagnostics={};scale=None;width=None;noise=None
        if samples:
            scale=median(g['torso'] for _,_,g in samples)
            width=median(g['width'] for _,_,g in samples)
            scale_range=(max(g['torso'] for _,_,g in samples)-min(g['torso'] for _,_,g in samples))/scale
            xs=[g['hip'][0] for _,_,g in samples];ys=[g['hip'][1] for _,_,g in samples]
            drift=math.hypot(max(xs)-min(xs),max(ys)-min(ys))/scale
            a0=samples[0][2]['angle']
            angles=[(g['angle']-a0+180)%360-180 for _,_,g in samples]
            angle_range=max(angles)-min(angles)
            diagnostics={'torso_range_ratio':scale_range,'hip_range_torso_ratio':drift,'shoulder_angle_range_deg':angle_range}
            if scale_range>.1:reasons.append('unstable_baseline_scale')
            if drift>.03:reasons.append('baseline_motion')
            if angle_range>8:reasons.append('baseline_rotation')
            noise=1.4826*median(abs(y-median(ys)) for y in ys)/scale
        valid=not reasons
        result['views'][view]={'valid':valid,'reasons':sorted(set(reasons)),
            'source_frames':[i for i,_,_ in samples],'window_frames':[r['frame_id'] for r,_ in window],
            'coverage':coverage,'sample_count':len(samples),'time_basis':basis,
            'scale_source':'baseline_median_projected_torso_length',
            'scale_px':scale if valid else None,'shoulder_width_baseline_px':width if valid else None,
            'hip_position_scaled_mad_torso_ratio':noise if valid else None,
            'diagnostics':diagnostics,'limitations':['stationarity_heuristic_not_camera_motion_compensated',
            'position_dispersion_not_displacement_uncertainty','projected_torso_is_not_constant_physical_scale']}
    return result


def normalized_view_trends(frames, profiles, start, end):
    result={}
    for view, baseline in profiles['views'].items():
        series=[]
        if baseline['valid']:
            for row in frames:
                if not start<=row['frame_id']<=end:continue
                g=_geometry(row,view);t,b=source_timestamp(row)
                if g is None or t is None or b!=baseline['time_basis']:continue
                series.append({'frame_id':row['frame_id'],'source_seconds':t,
                    'shoulder_projection_ratio':g['width']/baseline['shoulder_width_baseline_px'],
                    'torso_scale_ratio':g['torso']/baseline['scale_px']})
        result[view]={'valid':baseline['valid'],'reasons':baseline['reasons'],
                      'series':series,'coach_eligible':False,'accuracy_validated':False}
    return result
