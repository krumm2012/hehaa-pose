"""Public consumers must not turn unvalidated observations into teaching."""
import unittest
from local_realtime_coach import LocalRealtimeCoach
from coach_evidence_policy import build_coach_decision_policy
from swing_coach_data_collector import build_coach_dataset
from test_event_source_timing import source_documents


class CoachRuleEligibilityTests(unittest.TestCase):
    def test_dataset_does_not_score_pixel_speed_or_classify_projected_turn_quality(self):
        frames, analysis = source_documents()
        for feature in analysis['features']:
            feature.update(racket_speed=6., racket_center=[100., 200.])
        analysis['events'][0]['biomechanics'] = {'metrics': {
            'shoulder_turn_change': {'value': 3., 'coach_eligible': True}}}
        event = build_coach_dataset(frames, analysis)['events'][0]
        self.assertEqual(event['racket']['max_racket_speed'], 6.)
        self.assertIsNone(event['scores']['racket_speed_score'])
        self.assertEqual(event['body']['unit_turn_quality'], 'unknown')
        self.assertNotIn('low_racket_speed', event['diagnosis_tags'])

    def test_external_flags_and_legacy_score_cannot_approve_local_rules(self):
        metrics = {name: {'value': 1., 'confidence': .99, 'coach_eligible': True,
                         'accuracy_validated': True, 'unit': 'deg (2D)'}
                   for name in ('arm_extension', 'preparation_knee_flexion',
                                'hip_shoulder_separation', 'shoulder_turn_change',
                                'leg_drive', 'takeback_depth', 'brush_angle')}
        metrics['kinematic_sequence'] = {'value': 'DISCONNECTED', 'confidence': .99,
                                       'coach_eligible': True}
        event = {'confidence': .99, 'quality_flags': {'warnings': []},
                 'biomechanics': {'metrics': metrics},
                 'practice_score': {'score': 99., 'method': 'automatic_2d_projection'}}
        advice = LocalRealtimeCoach().advise_all(event)
        self.assertTrue(advice)
        self.assertTrue(all(row['category'] in ('review', 'capture') for row in advice))

    def test_legacy_derived_labels_do_not_authorize_language_model_coaching(self):
        metrics = {'confidence': .99, 'body': {'unit_turn_quality': 'limited',
                  'late_contact': True, 'contact_too_close_to_body': True,
                  'balance_state': 'unstable'}, 'scores': {'power_transfer_score': .1},
                  'coach_calibration': {'assessments': {'arm_extension':
                    {'status': 'usable', 'value': 1., 'coach_eligible': True}}}}
        policy = build_coach_decision_policy({'warnings': [], 'pose_frame_ratio': 1.}, metrics, {})
        self.assertFalse(policy['coaching_allowed'])
        self.assertEqual(policy['advice_candidates'], [])
        self.assertNotIn('technique', policy['allowed_advice_categories'])
        self.assertNotIn('positive', policy['allowed_advice_categories'])


if __name__ == '__main__':
    unittest.main()
