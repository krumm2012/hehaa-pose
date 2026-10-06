"""Short but monotone PTS must not silently authorize unstable motion peaks."""
import copy
import unittest
from test_kinematic_cross_validation import frames
from kinematic_sequence import analyze_kinematic_sequence


class CadenceAuditTests(unittest.TestCase):
    def test_short_interval_peak_is_audited_without_rewriting_input(self):
        rows = frames()
        # Compress two intervals at the actual hip pulse; timestamps stay monotone.
        times = [0.0]
        for i in range(1, len(rows)):
            times.append(times[-1] + (.0025 if i in (11, 12) else .04))
        for row, t in zip(rows, times):
            row['timestamp'] = t
        original = copy.deepcopy(rows)
        result = analyze_kinematic_sequence(rows, 20, 25)
        self.assertEqual(rows, original)
        self.assertEqual(result['cadence_audit']['short_interval_source_frames'], [11, 12])
        hip = result['views']['front']['segments']['hip']
        self.assertEqual(hip['status'], 'cadence_sensitive_peak')
        self.assertIsNone(hip['peak'])
        self.assertIsNone(result['is_sequential'])
        self.assertEqual(result['cross_validation']['reason'], 'cadence_sensitive_peak')
        self.assertIsNotNone(result.get('candidate_latency_hip_to_shoulder_ms'))
        self.assertIsNotNone(result.get('candidate_hip_peak_frame'))

    def test_regular_cadence_preserves_known_order(self):
        result = analyze_kinematic_sequence(frames(), 20, 25)
        self.assertEqual(result['cadence_audit']['short_interval_source_frames'], [])
        self.assertEqual(result['sequence_quality'], 'PROJECTED_ORDER')

    def test_short_interval_away_from_peaks_does_not_blanket_reject_vfr(self):
        rows = frames()
        for i in range(23, len(rows)):
            rows[i]['timestamp'] -= .0375
        result = analyze_kinematic_sequence(rows, 20, 25)
        self.assertEqual(result['cadence_audit']['short_interval_source_frames'], [23])
        self.assertEqual(result['sequence_quality'], 'PROJECTED_ORDER')
        self.assertTrue(result['views']['front']['segments']['hip']['cadence_sensitivity']['stable'])

    def test_rejected_back_joints_are_explained(self):
        result = analyze_kinematic_sequence(frames(back_confidence=.1), 20, 25)
        hip = result['views']['back']['segments']['hip']
        self.assertGreater(hip['observation_rejections']['low_joint_score'], 0)
        self.assertEqual(hip['observation_valid_frames'], 0)

    def test_report_shows_time_audit_and_back_rejections(self):
        from swing_report_builder import _build_kinematic_sequence_html
        page = _build_kinematic_sequence_html(analyze_kinematic_sequence(frames(back_confidence=.1), 20, 25))
        self.assertIn('时间间隔审计', page)
        self.assertIn('关节点分数不足', page)


if __name__ == '__main__':
    unittest.main()
