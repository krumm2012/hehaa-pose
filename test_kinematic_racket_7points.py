import unittest
import copy
import math

from kinematic_sequence import _racket_evidence
from joint_racket_2d_analysis import associate_rackets, analyze_frames, BODY_POINTS, RACKET_POINTS


def make_kpt(x, y, conf=0.9, fid=0):
    return {'x': float(x), 'y': float(y), 'confidence': float(conf), 'observed': True, 'source_frame_id': fid}


class KinematicRacket7PointsTests(unittest.TestCase):
    def test_7points_sweet_spot_preferred_over_box_center(self):
        # 10 frames where box center is stationary at (100, 100),
        # but 7-point sweet_spot moves with increasing speed simulating impact acceleration.
        rows = []
        times = [i * 0.04 for i in range(10)]
        for i in range(10):
            # Box center remains stationary at (100, 100)
            box = [90.0, 90.0, 110.0, 110.0]
            # Sweet spot accelerates along X
            sx = 200.0 + i * (i * 5.0)
            candidate = {
                'source': 'racket_pose_model',
                'source_frame_id': i,
                'observed': True,
                'confidence': 0.95,
                'keypoints': {
                    'butt_cap': make_kpt(sx - 50, 100, fid=i),
                    'grip_center': make_kpt(sx - 35, 100, fid=i),
                    'throat': make_kpt(sx - 15, 100, fid=i),
                    'sweet_spot': make_kpt(sx, 100, fid=i),
                    'head_top': make_kpt(sx + 20, 100, fid=i),
                    'frame_left': make_kpt(sx, 80, fid=i),
                    'frame_right': make_kpt(sx, 120, fid=i),
                }
            }
            row = {
                'frame_id': i,
                'rackets': [{'box': box, 'confidence': 0.9, 'observed': True, 'source_frame_id': i}],
                'racket_pose': {
                    'frame_id': i,
                    'status': 'observed',
                    'coordinate_space': 'original_source_pixels',
                    'candidates': [candidate]
                }
            }
            rows.append(row)

        evidence = _racket_evidence(rows, times, 0.04)
        self.assertEqual(evidence.get('point_source'), 'sweet_spot')
        # Since sweet spot is accelerating, its candidate peak speed should be substantially > 0
        cand_peak = evidence.get('candidate_peak')
        self.assertIsNotNone(cand_peak)
        self.assertGreater(cand_peak['speed'], 100.0)

    def test_fallback_to_box_center_when_racket_pose_absent(self):
        rows = []
        times = [i * 0.04 for i in range(10)]
        for i in range(10):
            cx = 100.0 + i * 15.0
            box = [cx - 10, 90.0, cx + 10, 110.0]
            row = {
                'frame_id': i,
                'rackets': [{'box': box, 'confidence': 0.9, 'observed': True, 'source_frame_id': i}]
            }
            rows.append(row)

        evidence = _racket_evidence(rows, times, 0.04)
        self.assertEqual(evidence.get('point_source'), 'box_center')
        self.assertIsNotNone(evidence.get('candidate_peak'))


class ForearmRacketAssociationTests(unittest.TestCase):
    def test_forearm_collinearity_resolves_grip_ambiguity(self):
        # Two racket candidates with almost identical distance to wrist (distance diff < margin),
        # but Candidate 0 extends in the natural forearm direction, while Candidate 1 is reverse.
        size = (1280, 720)
        # Forearm: elbow at (100, 200), wrist at (160, 200) -> arm vector is (+60, 0)
        bodies = {
            'front': {
                'right_elbow': [100.0, 200.0],
                'right_wrist': [160.0, 200.0],
                'left_elbow': None,
                'left_wrist': None,
            },
            'back': {name: None for name in BODY_POINTS}
        }
        # Candidate 0: grip at (175, 200) (dist=15px). Throat at (210, 200) -> racket vector is (+35, 0), cos=1.0
        c0 = {
            'source': 'racket_pose_model', 'source_frame_id': 0, 'observed': True, 'confidence': 0.9,
            'keypoints': {
                'grip_center': make_kpt(175, 200),
                'throat': make_kpt(210, 200),
                'sweet_spot': make_kpt(230, 200),
                'butt_cap': make_kpt(165, 200),
                'head_top': make_kpt(250, 200),
                'frame_left': make_kpt(230, 185),
                'frame_right': make_kpt(230, 215),
            }
        }
        # Candidate 1: grip at (176, 200) (dist=16px, diff=1px < margin which is ~7px).
        # But throat at (140, 200) -> racket vector is (-36, 0), cos = -1.0 (reversed!)
        c1 = {
            'source': 'racket_pose_model', 'source_frame_id': 0, 'observed': True, 'confidence': 0.9,
            'keypoints': {
                'grip_center': make_kpt(176, 200),
                'throat': make_kpt(140, 200),
                'sweet_spot': make_kpt(120, 200),
                'butt_cap': make_kpt(186, 200),
                'head_top': make_kpt(100, 200),
                'frame_left': make_kpt(120, 185),
                'frame_right': make_kpt(120, 215),
            }
        }
        row = {
            'frame_id': 0,
            'racket_pose': {
                'frame_id': 0,
                'status': 'observed',
                'coordinate_space': 'original_source_pixels',
                'candidates': [c1, c0]  # c1 is listed first, but c0 has forearm collinearity!
            }
        }
        res = associate_rackets(row, bodies, size)
        # Because c0 has alignment 1.0 and c1 has alignment -1.0, tie-breaker should choose c0 (candidate_index 1)
        self.assertEqual(res['front']['status'], 'proximity_candidate')
        self.assertEqual(res['front']['candidate_index'], 1)

    def test_identical_candidates_remain_ambiguous(self):
        size = (1280, 720)
        bodies = {
            'front': {
                'right_elbow': [100.0, 200.0],
                'right_wrist': [160.0, 200.0],
                'left_elbow': None,
                'left_wrist': None,
            },
            'back': {name: None for name in BODY_POINTS}
        }
        c0 = {
            'source': 'racket_pose_model', 'source_frame_id': 0, 'observed': True, 'confidence': 0.9,
            'keypoints': {
                'grip_center': make_kpt(175, 200),
                'throat': make_kpt(210, 200),
                'sweet_spot': make_kpt(230, 200),
                'butt_cap': make_kpt(165, 200),
                'head_top': make_kpt(250, 200),
                'frame_left': make_kpt(230, 185),
                'frame_right': make_kpt(230, 215),
            }
        }
        # c1 has exact same position and alignment
        c1 = copy.deepcopy(c0)
        row = {
            'frame_id': 0,
            'racket_pose': {
                'frame_id': 0,
                'status': 'observed',
                'coordinate_space': 'original_source_pixels',
                'candidates': [c0, c1]
            }
        }
        res = associate_rackets(row, bodies, size)
        self.assertEqual(res['front']['status'], 'unavailable')
        self.assertEqual(res['front']['reason'], 'ambiguous_grip_assignment')


if __name__ == '__main__':
    unittest.main()
