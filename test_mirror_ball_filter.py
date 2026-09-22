"""Safety regressions for camera-specific mirror suppression."""
import copy
import unittest
from collections import deque

import numpy as np

from ball_track_selector import BallTrackSelector
from mirror_ball_filter import MirrorBallFilter


def ball(x, y, size=8, confidence=.9):
    return {"position": [x, y], "box": [x-size/2, y-size/2, x+size/2, y+size/2], "confidence": confidence}


def config(mode="enforce"):
    return {"mode": mode, "polygon": [[.1,.1],[.9,.1],[.9,.6],[.1,.6]],
            "max_size_height_ratio": .02, "boundary_margin_height_ratio": .01}


class MirrorBallFilterTests(unittest.TestCase):
    def assess(self, candidate, **context):
        return MirrorBallFilter(config()).assess(candidate, frame_width=1000, frame_height=1000, **context)

    def test_small_mirror_ball_requires_both_location_and_size(self):
        self.assertTrue(self.assess(ball(500,300))["rejected"])
        self.assertFalse(self.assess(ball(500,800))["rejected"])
        self.assertFalse(self.assess(ball(500,300,size=30))["rejected"])

    def test_boundary_band_and_elongated_blur_are_kept(self):
        self.assertFalse(self.assess(ball(500,595))["rejected"])
        elongated=ball(500,300)
        elongated['box']=[480,297,520,303]
        self.assertFalse(self.assess(elongated)["rejected"])

    def test_missing_invalid_geometry_and_dimensions_fail_open(self):
        for box in [None, [], [1,2,1,4], [1,2,float('nan'),4], ['bad',2,3,4]]:
            candidate=ball(500,300);candidate['box']=box
            self.assertFalse(self.assess(candidate)["rejected"])
        result=MirrorBallFilter(config()).assess(ball(500,300),frame_width=None,frame_height=1000)
        self.assertFalse(result['rejected'])

    def test_scale_invariance(self):
        for scale in [.5,1.,2.]:
            result=MirrorBallFilter(config()).assess(ball(500*scale,300*scale,8*scale),frame_width=1000*scale,frame_height=1000*scale)
            self.assertTrue(result['rejected'])

    def test_trajectory_and_occlusion_prediction_protect_small_real_ball(self):
        self.assertEqual(self.assess(ball(500,300),previous_position=[490,310])['reason'],'trajectory_protected')
        result=self.assess(ball(700,300),previous_position=[400,300],previous_velocity=[100,0],missed_frames=2,continuity_distance=60,prediction_distance=20)
        self.assertEqual(result['reason'],'prediction_protected')
        self.assertFalse(result['rejected'])

    def test_reflected_racket_does_not_exempt_mirror_ball(self):
        self.assertTrue(self.assess(ball(500,300),racket_centers=[[500,320]])['rejected'])
        self.assertFalse(self.assess(ball(500,570),racket_centers=[[500,620]])['rejected'])

    def test_shadow_reports_without_rejecting(self):
        result=MirrorBallFilter(config('shadow')).assess(ball(500,300),frame_width=1000,frame_height=1000)
        self.assertTrue(result['would_reject'])
        self.assertFalse(result['rejected'])

    def test_disabled_needs_no_camera_calibration(self):
        self.assertFalse(MirrorBallFilter().assess(ball(500,300),frame_width=1000,frame_height=1000)['rejected'])

    def test_invalid_calibration_is_explicit(self):
        for settings in [dict(mode='bad'),dict(mode='enforce',polygon=[]),{**config(),'max_size_height_ratio':float('nan')}]:
            with self.assertRaises(ValueError):MirrorBallFilter(settings)


class MirrorTrackIntegrationTests(unittest.TestCase):
    def selector(self, mode='enforce', **extra):
        return BallTrackSelector({'mirror_ball_filter':config(mode),'static_ball_suppression_enabled':False,**extra})

    def select(self, selector, balls, rackets=None):
        return selector.select(balls,rackets,frame_height=1000,frame_width=1000)

    def test_prevents_reflected_track_seed_with_real_ball_present(self):
        selector=self.selector()
        mirror=ball(500,300,confidence=.99)
        real=ball(500,800,size=30,confidence=.7)
        result=self.select(selector,[mirror,real])
        self.assertEqual(result.active_ball['position'],real['position'])
        self.assertEqual(result.diagnostics['rejections']['small_mirror_unsupported'],1)

    def test_real_trajectory_crosses_mirror_and_turns_without_new_filter_drop(self):
        selector=self.selector(active_ball_motion_filter_enabled=True)
        for x,y,size in [(500,680,30),(500,620,25),(500,560,8),(500,500,8),(510,480,8),(520,510,8)]:
            real=ball(x,y,size)
            result=self.select(selector,[real,ball(750,300,confidence=.99)])
            self.assertIsNotNone(result.active_ball)
            self.assertEqual(result.active_ball['position'],real['position'])

    def test_fast_track_is_protected_after_missing_frames(self):
        selector=self.selector(active_ball_motion_filter_enabled=True)
        self.select(selector,[ball(200,650,30)])
        self.select(selector,[ball(300,550,30)])
        self.select(selector,[])
        self.select(selector,[])
        result=self.select(selector,[ball(600,250,8)])
        self.assertEqual(result.active_ball['position'],[600,250])
        self.assertEqual(result.diagnostics['mirror_filter']['candidates'][0]['reason'],'prediction_protected')

    def test_expired_trajectory_does_not_protect_new_mirror_track(self):
        selector=self.selector(active_ball_reacquisition_frames=2)
        self.select(selector,[ball(500,300,30)])
        for _ in range(3):self.select(selector,[])
        self.assertIsNone(self.select(selector,[ball(500,300,8)]).active_ball)

    def test_shadow_and_off_have_identical_selection_and_inputs_unchanged(self):
        shadow,off=self.selector('shadow'),self.selector('off')
        sequence=[[ball(500+i*10,300),ball(800,800,30,.75)] for i in range(5)]
        untouched=copy.deepcopy(sequence)
        for candidates in sequence:
            self.assertEqual(self.select(shadow,candidates).active_ball,self.select(off,candidates).active_ball)
        self.assertEqual(sequence,untouched)

    def test_unseen_small_real_ball_inside_mirror_is_known_ambiguity(self):
        # Region + apparent size cannot identify an untracked foreground ball.
        # Keep shadow as rollout default until this case has labeled coverage.
        result=self.select(self.selector('shadow'),[ball(500,300)])
        self.assertIsNotNone(result.active_ball)
        self.assertTrue(result.diagnostics['mirror_filter']['candidates'][0]['would_reject'])

    def test_detector_passes_full_frame_dimensions_after_roi_offset(self):
        from yolo26n_unified_detector import YOLO26nUnifiedDetector
        detector = YOLO26nUnifiedDetector.__new__(YOLO26nUnifiedDetector)
        detector.config = {"overlay_marker_recovery_enabled": False}
        detector.model = type("FakeModel", (), {"predict": lambda self, inputs: {}})()
        detector._coreml_input_names = set()
        detector._preprocess = lambda frame: frame
        detector._parse_predictions = lambda predictions: ([ball(100, 100)], [])
        detector.detection_times = deque(maxlen=10)
        detector.ball_track_selector = self.selector()
        detector._select_primary_racket = lambda rackets, balls: []
        balls, _, _ = detector.detect_unified(
            np.zeros((200, 200, 3), dtype=np.uint8),
            coordinate_offset=(400, 200), full_frame_size=(1000, 1000),
        )
        self.assertEqual(balls, [])
        evidence = detector.last_ball_diagnostics['mirror_filter']['candidates'][0]
        self.assertEqual(evidence['position'], [500, 300])
        self.assertTrue(evidence['rejected'])


if __name__=='__main__':unittest.main()
