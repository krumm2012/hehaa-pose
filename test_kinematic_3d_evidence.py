import unittest

import numpy as np

from scripts.validate_kinematic_3d_evidence import direction_speed, segment_audit


class EvidenceAuditTests(unittest.TestCase):
    def test_reflection_recovers_same_anatomical_segment(self):
        angles = np.linspace(0, 1, 12)
        front = np.zeros((12, 11, 3))
        front[:, 10] = np.stack([np.cos(angles), np.zeros(12), np.sin(angles)], axis=1)
        mirror = front.copy()
        mirror[:, :, 2] *= -1
        report = segment_audit(front, mirror, np.ones(12, dtype=bool), [0, 0, 1], (9, 10), 25)
        self.assertLess(report['direction_difference_deg']['p90'], .02)
        self.assertLess(report['speed_difference_deg_per_s']['p90'], .02)

    def test_missing_frame_does_not_bridge_derivatives(self):
        units = np.array([[1., 0, 0], [0, 0, 1], [-1, 0, 0], [0, 0, -1]])
        speed = direction_speed(units, np.array([True, False, True, True]), 25)
        self.assertTrue(np.isnan(speed[:3]).all())
        self.assertAlmostEqual(speed[3], 2250)

    def test_zero_length_joint_pair_abstains(self):
        data = np.zeros((12, 11, 3))
        report = segment_audit(data, data, np.ones(12, dtype=bool), [0, 0, 1], (9, 10), 25)
        self.assertEqual(report['valid_frames'], 0)
        self.assertIsNone(report['direction_difference_deg']['median'])
        self.assertIsNone(report['same_frame_direction_speed_correlation'])


if __name__ == '__main__':
    unittest.main()
