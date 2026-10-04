"""Human-review validation and direct derivation share strict source identities."""
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from manual_review_workflow import (derive_manual_coach_events, discover_session_paths,
                                    process_manual_review, validate_manual_annotations)
from test_manual_review_workflow import _annotations, _event_document, _frame


class ManualReviewIdentityContractTests(unittest.TestCase):
    def test_validation_rejects_model_and_link_identifiers_without_coercion(self):
        for side in ('model', 'reference'):
            for value in (1.9, True, '1'):
                with self.subTest(side=side, value=value):
                    model, reference = _event_document(), _annotations(False)
                    row = model['events'][0] if side == 'model' else reference['events'][0]
                    row['event_id' if side == 'model' else 'source_event_id'] = value
                    with self.assertRaises(ValueError):
                        validate_manual_annotations(model, reference, [_frame(i) for i in range(20)])

    def test_validation_rejects_invalid_declared_journal_identities(self):
        for value in (8.9, True, '8', -1, 2**53):
            with self.subTest(value=value):
                records = [_frame(i) for i in range(20)]
                records[8]['frame_id'] = value
                with self.assertRaises(ValueError):
                    validate_manual_annotations(_event_document(), _annotations(False), records)

    def test_direct_derivation_rejects_wrong_identity_before_feature_extraction(self):
        reference = _annotations(False)
        reference['events'][0]['source_event_id'] = 1.9
        with patch('manual_review_workflow.extract_motion_features') as extraction:
            with self.assertRaises(ValueError):
                derive_manual_coach_events(_event_document(), reference, [_frame(i) for i in range(20)])
            extraction.assert_not_called()

    def test_invalid_journal_does_not_replace_previous_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'events.json').write_text(json.dumps(_event_document()))
            (root / 'frames.jsonl').write_text(''.join(json.dumps(_frame(i)) + '\n' for i in range(20)))
            (root / 'report.html').write_text('review')
            paths = discover_session_paths(root)
            reference = _annotations(True)
            reference['source']['event_json'] = 'events.json'
            process_manual_review(paths, reference)
            keys = ('events', 'annotations', 'evaluation', 'state')
            before = {key: paths[key].read_bytes() for key in keys}
            records = [_frame(i) for i in range(20)]
            records[8]['frame_id'] = 8.9
            paths['frames'].write_text(''.join(json.dumps(row) + '\n' for row in records))
            with self.assertRaises(ValueError):
                process_manual_review(paths, reference)
            self.assertEqual({key: paths[key].read_bytes() for key in keys}, before)

    def test_valid_reference_zero_and_integer_identity_remain_unmodified(self):
        model, reference = _event_document(), _annotations(False)
        records = [_frame(i) for i in range(20)]
        reference['events'][0]['frames']['start'] = 0
        original = deepcopy((model, reference, records))
        result = derive_manual_coach_events(model, reference, records)
        self.assertEqual(result['events'][0]['start_frame'], 0)
        self.assertEqual(result['events'][0]['contact_frame'], 9)
        self.assertEqual(result['events'][0]['source_event_id'], 1)
        self.assertEqual((model, reference, records), original)


if __name__ == '__main__':
    unittest.main()
