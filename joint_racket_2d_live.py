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

    def draw_source(self, canvas):
        import cv2
        def xy(p): return tuple(int(round(v)) for v in p)
        for key, a, b in self.segments():
            color = (255, 190, 70) if '.body.' in key else (80, 220, 220)
            cv2.line(canvas, xy(a), xy(b), color, 2, cv2.LINE_AA)
        if not self.history:
            return
        tracks = self.history[-1]['tracks']
        for a, b in LINKS:
            if tracks[a] is not None and tracks[b] is not None:
                cv2.line(canvas, xy(tracks[a]), xy(tracks[b]), (90, 220, 140), 2, cv2.LINE_AA)
        for key in set(k for pair in LINKS for k in pair):
            if tracks[key] is not None:
                cv2.circle(canvas, xy(tracks[key]), 4, (90, 220, 140), -1, cv2.LINE_AA)

    @staticmethod
    def draw_status(canvas, diagnostic):
        import cv2
        gaps = diagnostic['active_gaps_ge_3']
        short = lambda g: g['track'].removeprefix('front.').replace('complete.', '')
        message = ('2D FRONT | candidate links | ' +
                   ('GAP >=3: ' + ', '.join(f"{short(g)}:{g['count']}" for g in gaps[:3])
                    if gaps else 'no active gap >=3'))
        # Wrap to the rendered width; ASCII is supported by the OpenCV font.
        chars = max(20, int(canvas.shape[1] / 8))
        lines = [message[i:i+chars] for i in range(0, len(message), chars)]
        height = len(lines) * 20 + 8
        cv2.rectangle(canvas, (0, max(0, canvas.shape[0]-height)),
                      (canvas.shape[1], canvas.shape[0]), (20, 20, 20), -1)
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (8, canvas.shape[0]-height+19+i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (100, 220, 240), 1, cv2.LINE_AA)
