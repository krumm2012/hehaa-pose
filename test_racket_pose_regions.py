import copy
import unittest

import numpy as np

from racket_pose_regions import configured_regions, restore_source_observation
from racket_pose_detector import decode_candidates, letterbox
from scripts.compare_racket_regions import merge_observations


class RacketPoseRegionTests(unittest.TestCase):
    front = {'default_roi': [.45, .08, .75, .85]}
    mirror = {'reflection_roi': [.4449, 0, .8094, .3229]}

    def test_full_needs_no_camera_configuration(self):
        region, = configured_regions((2560, 1440), {}, {}, 'full')
        self.assertEqual(region.box, (0, 0, 2560, 1440))

    def test_union_contains_both_expanded_views(self):
        split = configured_regions((2560, 1440), self.front, self.mirror, 'split', 256)
        union, = configured_regions((2560, 1440), self.front, self.mirror, 'union', 256)
        self.assertEqual(union.box, (882, 0, 2329, 1440))
        for r in split:
            self.assertTrue(union.box[0] <= r.box[0] < r.box[2] <= union.box[2])
            self.assertTrue(union.box[1] <= r.box[1] < r.box[3] <= union.box[3])

    def test_extract_uses_original_pixels_and_no_flip(self):
        frame = np.arange(20 * 30 * 3).reshape(20, 30, 3)
        region, = configured_regions((30, 20), {'default_roi': [.2, .1, .8, .9]},
                                     {'reflection_roi': [.3, 0, .6, .3]}, 'union', 0)
        x, y, right, bottom = region.box
        np.testing.assert_array_equal(region.extract(frame), frame[y:bottom, x:right])
        self.assertTrue(np.shares_memory(region.extract(frame), frame))

    def test_restore_preserves_source_identity_and_raw_measurement_provenance(self):
        region = configured_regions((2560, 1440), self.front, self.mirror, 'split', 0)[0]
        original = {'frame_id': 17, 'source_time': {'timestamp_seconds': .68},
                    'candidates': [{'box': [2, 3, 22, 33], 'source_frame_id': 17,
                                    'observed': True, 'measurement_eligible': False,
                                    'keypoints': {str(i): {'x': 5., 'y': 8., 'observed': i != 0,
                                                          'confidence': .9, 'source_frame_id': 17}
                                                  for i in range(7)}}]}
        saved = copy.deepcopy(original)
        restored = restore_source_observation(original, region, (2560, 1440))
        self.assertEqual(original, saved)
        self.assertEqual(restored['candidates'][0]['box'], [1154, 118, 1174, 148])
        for p in restored['candidates'][0]['keypoints'].values():
            self.assertEqual((p['x'], p['y'], p['source_frame_id']), (1157., 123., 17))
        self.assertFalse(restored['candidates'][0]['keypoints']['0']['observed'])
        self.assertFalse(restored['candidates'][0]['measurement_eligible'])
        self.assertEqual(restored['source_time'], original['source_time'])
        self.assertNotIn('association_view', restored['candidates'][0])

    def test_invalid_configuration_never_silently_becomes_an_accepted_crop(self):
        for value in (None, [0, 0, 0, 1], [0, 0, 2, 1], [0, 0, float('nan'), 1]):
            with self.assertRaises(ValueError):
                configured_regions((2560, 1440), {'default_roi': value}, self.mirror, 'union')
        for margin in (-1, .5, True):
            with self.assertRaises(ValueError):
                configured_regions((2560, 1440), self.front, self.mirror, 'union', margin)

    def test_actual_letterbox_decode_then_source_offset_for_all_seven_points(self):
        region = configured_regions((2560, 1440), self.front, self.mirror, 'split', 0)[1]
        width, height = region.size
        _, transform = letterbox(np.zeros((height, width, 3), np.uint8))
        raw = np.zeros((1, 26, 18900), np.float32)
        gain, left, top = (transform[k] for k in ('gain', 'left', 'top'))
        raw[0, :5, 0] = [200 * gain + left, 150 * gain + top, 100 * gain, 120 * gain, .9]
        for i in range(7):
            raw[0, 5 + i * 3:8 + i * 3, 0] = [(170 + i * 5) * gain + left,
                                               (110 + i * 10) * gain + top, .95]
        candidates, diag = decode_candidates(raw, transform, {'confidence_threshold': .25,
                                                              'keypoint_threshold': .5, 'iou_threshold': .45})
        observed = restore_source_observation({'candidates': candidates, 'diagnostics': diag}, region, (2560, 1440))
        candidate = observed['candidates'][0]
        np.testing.assert_allclose(candidate['box'], [region.box[0] + 150, 90, region.box[0] + 250, 210], atol=.001)
        for i, point in enumerate(candidate['keypoints'].values()):
            np.testing.assert_allclose([point['x'], point['y']], [region.box[0] + 170 + i * 5, 110 + i * 10], atol=.001)
            self.assertTrue(point['observed'])

    def test_cross_region_nms_preserves_full_source_candidate_and_numerical_failure(self):
        def part(confidence, nonfinite=0):
            return {'frame_id': 9, 'status': 'observed', 'input_region': {'name': 'example'},
                    'candidates': [{'box': [100, 50, 140, 90], 'confidence': confidence,
                                    'source_frame_id': 9}],
                    'diagnostics': {'candidates': 18900, 'nonfinite_candidates': nonfinite,
                                    'nonfinite_values': nonfinite}}
        a, b = part(.8), part(.9, 1)
        merged = merge_observations([a, b], .25, .45)
        self.assertEqual(len(merged['candidates']), 1)
        self.assertEqual(merged['candidates'][0]['confidence'], .9)
        self.assertEqual(merged['candidates'][0]['source_frame_id'], 9)
        self.assertEqual(merged['status'], 'numerical_failure')
        self.assertEqual(len(a['candidates']), 1)


if __name__ == '__main__':
    unittest.main()
