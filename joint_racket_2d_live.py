"""Bounded source-space 2D trails for the existing rendered live preview."""
from collections import deque
from joint_racket_2d_analysis import analyze_frames

TRAIL_KEYS = ('front.body.right_wrist', 'front.racket.grip_center',
              'front.racket.sweet_spot', 'front.racket.head_top')
LINKS = (('front.body.right_shoulder', 'front.body.right_elbow'),
         ('front.body.right_elbow', 'front.body.right_wrist'),
         ('front.body.right_wrist', 'front.racket.grip_center'),
         ('front.racket.grip_center', 'front.racket.sweet_spot'),
         ('front.racket.sweet_spot', 'front.racket.head_top'))


class LiveJointRacket2D:
    def __init__(self, history_frames=24):
        self.history = deque(maxlen=history_frames)
        self.gaps = {}
        self.previous_id = None

    def update(self, row, image_size):
        fid = row['frame_id']
        discontinuity = self.previous_id is not None and fid != self.previous_id + 1
        if discontinuity:
            self.history.clear(); self.gaps.clear()
        frame = analyze_frames([row], image_size)[0][0]
        active = []
        keys = tuple(k for k in frame['tracks'] if k.startswith('front.'))
        for key in keys:
            if frame['tracks'][key] is None:
                gap = self.gaps.setdefault(key, {'track': key, 'start_frame': fid, 'count': 0})
                gap['count'] += 1; gap['end_frame'] = fid
                gap['reason'] = frame['missing_reasons'][key]
                if gap['count'] >= 3:
                    active.append(dict(gap))
            else:
                self.gaps.pop(key, None)
        self.history.append(frame)
        self.previous_id = fid
        return {'policy': 'joint_racket_2d_live_v1', 'frame_id': fid,
                'view': 'front', 'coordinate_space': 'original_source_pixels',
                'assignment': frame['associations']['front'],
                'active_gaps_ge_3': active, 'source_discontinuity': discontinuity,
                'history_frames': len(self.history), 'coach_eligible': False}

    def segments(self):
        result = []
        for before, after in zip(self.history, list(self.history)[1:]):
            for key in TRAIL_KEYS:
                a, b = before['tracks'][key], after['tracks'][key]
                if a is not None and b is not None and after['frame_id'] == before['frame_id'] + 1:
                    # A different proximity candidate is not a verified person track.
                    old = before['associations']['front'].get('candidate_index')
                    new = after['associations']['front'].get('candidate_index')
                    if '.body.' in key or old == new:
                        result.append((key, a, b))
        return result

    def draw_source(self, canvas, coord_mapper=None):
        import math
        import cv2

        def xy(p):
            if coord_mapper is not None:
                p = coord_mapper(p[0], p[1])
            return tuple(int(round(v)) for v in p)

        overlay = canvas.copy()
        segs = self.segments()
        total_segs = max(1, len(segs))
        for idx, (key, a, b) in enumerate(segs):
            # 过滤跨区域误匹配造成的长距离跳变伪轨迹线
            if math.hypot(b[0] - a[0], b[1] - a[1]) > 140.0:
                continue
            # 仅保留拍面甜区与手腕两条核心轨迹，避免 4 条平行黄青线糊满全身
            if key not in ('front.racket.sweet_spot', 'front.body.right_wrist'):
                continue
            fade = 0.35 + 0.65 * ((idx + 1) / total_segs)
            base_col = (235, 175, 65) if '.body.' in key else (70, 200, 210)
            color = tuple(int(round(c * fade)) for c in base_col)
            cv2.line(overlay, xy(a), xy(b), color, 1, cv2.LINE_AA)

        if self.history:
            tracks = self.history[-1]['tracks']
            for a, b in LINKS:
                if tracks[a] is not None and tracks[b] is not None:
                    if math.hypot(tracks[b][0] - tracks[a][0], tracks[b][1] - tracks[a][1]) <= 180.0:
                        cv2.line(overlay, xy(tracks[a]), xy(tracks[b]), (80, 200, 130), 1, cv2.LINE_AA)
            for key in set(k for pair in LINKS for k in pair):
                if tracks[key] is not None:
                    cv2.circle(overlay, xy(tracks[key]), 2, (80, 200, 130), -1, cv2.LINE_AA)

        # 低透明度叠加（0.28），让挥拍轨迹作为轻盈残影参考而不遮挡人物与球拍
        cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0, canvas)

    @staticmethod
    def draw_status(canvas, diagnostic):
        import cv2
        gaps = diagnostic['active_gaps_ge_3']
        short = lambda g: g['track'].removeprefix('front.').replace('complete.', '')
        message = ('2D FRONT | candidate links | ' +
                   ('GAP >=3: ' + ', '.join(f"{short(g)}:{g['count']}" for g in gaps[:3])
                    if gaps else 'no active gap >=3'))
        chars = max(20, int(canvas.shape[1] / 8))
        lines = [message[i:i+chars] for i in range(0, len(message), chars)]
        height = len(lines) * 16 + 6
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, max(0, canvas.shape[0]-height)),
                      (canvas.shape[1], canvas.shape[0]), (18, 20, 24), -1)
        cv2.addWeighted(overlay, 0.42, canvas, 0.58, 0, canvas)
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (8, canvas.shape[0]-height+14+i*16),
                        cv2.FONT_HERSHEY_SIMPLEX, .36, (130, 195, 210), 1, cv2.LINE_AA)
