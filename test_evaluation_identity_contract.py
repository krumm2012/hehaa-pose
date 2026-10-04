"""The public evaluation/writer seams must not invent source frame identities."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from swing_evaluation import evaluate_swing_events, write_evaluation_report
from manual_annotation_contract import MAX_SAFE_FRAME_ID


def evaluation_case():
    model = {'events': [{'event_id': 1, 'start_frame': 100, 'contact_frame': 110,
                         'end_frame': 120, 'stroke_type': 'Forehand'}]}
    reference = {'schema_version': 'swing_manual_annotations_v2',
                 'timeline_review_complete': True,
                 'events': [{'annotation_id': 'review-a', 'source_event_id': 1,
                             'actual_stroke_type': 'Forehand', 'valid_hit': True,
                             'frames': {'start': 100, 'contact': 110, 'end': 120}}]}
    return model, reference


class EvaluationIdentityContractTests(unittest.TestCase):
    def test_writer_rejects_fractional_reference_without_replacing_previous_report(self):
        model, reference = evaluation_case()
        reference['events'][0]['frames']['contact'] = 110.9
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            event_path = directory / 'events.json'
            reference_path = directory / 'annotations.json'
            output_path = directory / 'evaluation.json'
            event_path.write_text(json.dumps(model))
            reference_path.write_text(json.dumps(reference))
            output_path.write_text('previous evaluation')
            with self.assertRaises(ValueError):
                write_evaluation_report(str(event_path), str(reference_path), str(output_path))
            self.assertEqual(output_path.read_text(), 'previous evaluation')

    def test_public_evaluator_rejects_bool_model_frame(self):
        model, reference = evaluation_case()
        model['events'][0]['contact_frame'] = True
        with self.assertRaises(ValueError):
            evaluate_swing_events(model, reference)

    def test_both_schemas_reject_invalid_declared_model_and_reference_frames(self):
        for schema in ('swing_manual_annotations_v1', 'swing_manual_annotations_v2'):
            for value in (110.9, 110.0, True, '110', -1, MAX_SAFE_FRAME_ID + 1,
                          float('nan'), float('inf'), [], {}):
                for side in ('model', 'reference'):
                    with self.subTest(schema=schema, value=repr(value), side=side):
                        model, reference = evaluation_case()
                        reference['schema_version'] = schema
                        reference['events'][0]['event_id'] = 1
                        if side == 'model':
                            model['events'][0]['contact_frame'] = value
                        else:
                            reference['events'][0]['frames']['start'] = value
                        with self.assertRaises(ValueError):
                            evaluate_swing_events(model, reference)

    def test_invalid_identity_cannot_be_hidden_by_a_valid_compatibility_field(self):
        model, reference = evaluation_case()
        reference['events'][0]['frames']['contact'] = '110'
        reference['events'][0]['contact_frame'] = 110
        with self.assertRaises(ValueError):
            evaluate_swing_events(model, reference)

    def test_rejects_invalid_event_and_source_identifiers_without_coercion(self):
        for owner, key in (('model', 'event_id'), ('reference', 'event_id'),
                           ('reference', 'source_event_id')):
            for value in (True, 1.5, '1', -1, MAX_SAFE_FRAME_ID + 1):
                with self.subTest(owner=owner, key=key, value=value):
                    model, reference = evaluation_case()
                    target = model if owner == 'model' else reference
                    target['events'][0][key] = value
                    with self.assertRaises(ValueError):
                        evaluate_swing_events(model, reference)

    def test_duplicate_model_identity_is_rejected_in_both_schemas(self):
        for schema in ('swing_manual_annotations_v1', 'swing_manual_annotations_v2'):
            with self.subTest(schema=schema):
                model, reference = evaluation_case()
                reference['schema_version'] = schema
                reference['events'][0]['event_id'] = 1
                model['events'].append(deepcopy(model['events'][0]))
                with self.assertRaises(ValueError):
                    evaluate_swing_events(model, reference)

    def test_duplicate_reference_identity_is_rejected_in_both_schemas(self):
        for schema in ('swing_manual_annotations_v1', 'swing_manual_annotations_v2'):
            with self.subTest(schema=schema):
                model, reference = evaluation_case()
                reference['schema_version'] = schema
                reference['events'][0]['event_id'] = 1
                reference['events'].append(deepcopy(reference['events'][0]))
                with self.assertRaises(ValueError):
                    evaluate_swing_events(model, reference)

    def test_bad_containers_and_non_object_rows_are_not_silently_dropped(self):
        for side in ('model', 'reference'):
            for value in (None, {}, 'events', [None], [110]):
                with self.subTest(side=side, value=value):
                    model, reference = evaluation_case()
                    target = model if side == 'model' else reference
                    target['events'] = value
                    with self.assertRaises(ValueError):
                        evaluate_swing_events(model, reference)
            for value in ([], 'frames', False):
                with self.subTest(side=side, frames=value):
                    model, reference = evaluation_case()
                    target = model if side == 'model' else reference
                    target['events'][0]['frames'] = value
                    with self.assertRaises(ValueError):
                        evaluate_swing_events(model, reference)

    def test_null_reference_boundary_does_not_borrow_legacy_value(self):
        model, reference = evaluation_case()
        reference['events'][0]['frames']['start'] = None
        reference['events'][0]['start_frame'] = 100
        original = deepcopy((model, reference))
        report = evaluate_swing_events(model, reference)
        self.assertIsNone(report['events'][0]['manual_start_frame'])
        self.assertIsNone(report['events'][0]['start_delta_frames'])
        self.assertIsNone(report['events'][0]['event_iou'])
        self.assertEqual((model, reference), original)

    def test_null_model_contact_does_not_borrow_nested_value(self):
        model, reference = evaluation_case()
        model['events'][0]['contact_frame'] = None
        model['events'][0]['frames'] = {'contact': 110}
        report = evaluate_swing_events(model, reference)
        self.assertIsNone(report['events'][0]['model_contact_frame'])
        self.assertIsNone(report['events'][0]['contact_delta_frames'])
        self.assertEqual(report['summary']['contact_comparable'], 0)

    def test_zero_nested_model_contact_is_preserved(self):
        model, reference = evaluation_case()
        model['events'][0] = {'event_id': 0, 'stroke_type': 'Forehand',
                              'frames': {'start_frame': 0, 'contact_frame': 0,
                                         'contact': 2, 'end_frame': 3}}
        reference['events'][0]['frames'] = {'start': 0, 'contact': 0, 'end': 3}
        report = evaluate_swing_events(model, reference)
        self.assertEqual(report['events'][0]['model_event_id'], 0)
        self.assertEqual(report['events'][0]['model_contact_frame'], 0)
        self.assertEqual(report['events'][0]['contact_delta_frames'], 0)
        self.assertEqual(report['events'][0]['event_iou'], 1.)

    def test_inverted_or_out_of_interval_contact_is_not_repaired(self):
        for side in ('model', 'reference'):
            for start, contact, end in ((120, 110, 100), (100, 130, 120), (115, 110, 120)):
                with self.subTest(side=side, frames=(start, contact, end)):
                    model, reference = evaluation_case()
                    if side == 'model':
                        model['events'][0].update(start_frame=start, contact_frame=contact, end_frame=end)
                    else:
                        reference['events'][0]['frames'] = {'start': start, 'contact': contact, 'end': end}
                    with self.assertRaises(ValueError):
                        evaluate_swing_events(model, reference)

    def test_safe_large_frame_and_absent_compatibility_fields_keep_exact_deltas(self):
        model, reference = evaluation_case()
        base = MAX_SAFE_FRAME_ID - 20
        model['events'][0].update(start_frame=base, contact_frame=base + 10, end_frame=base + 20)
        reference['events'][0]['frames'] = {'start': base, 'contact': base + 11, 'end': base + 20}
        report = evaluate_swing_events(model, reference)
        self.assertEqual(report['events'][0]['contact_delta_frames'], -1)
        self.assertEqual(report['events'][0]['manual_end_frame'], MAX_SAFE_FRAME_ID)
        self.assertIsNone(report['summary']['contact_accuracy'])

    def test_missing_model_id_is_not_fabricated_from_row_index(self):
        model, reference = evaluation_case()
        del model['events'][0]['event_id']
        with self.assertRaises(ValueError):
            evaluate_swing_events(model, reference)

    def test_reference_ids_are_stable_before_invalid_hit_filtering(self):
        model, reference = evaluation_case()
        del reference['events'][0]['annotation_id']
        reference['events'].insert(0, {'valid_hit': False, 'actual_stroke_type': 'No Swing',
                                       'frames': {'start': 0, 'contact': 1, 'end': 2}})
        report = evaluate_swing_events(model, reference)
        self.assertEqual(report['events'][0]['annotation_id'], 'annotation-2')

    def test_frame_count_settings_are_not_truncated_or_clamped(self):
        for name in ('contact_tolerance_frames', 'match_contact_tolerance_frames'):
            for value in (True, 3.9, '3', -1):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        evaluate_swing_events(*evaluation_case(), **{name: value})

    def test_iou_setting_requires_finite_numeric_probability_range(self):
        for value in (True, '0.1', -0.1, 1.1, float('nan'), float('inf'), 10**1000):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                        evaluate_swing_events(*evaluation_case(), min_event_iou=value)

    def test_review_completion_and_hit_flags_cannot_use_truthy_strings(self):
        for field in ('timeline_review_complete', 'valid_hit', 'needs_review', 'count_correct'):
            with self.subTest(field=field):
                model, reference = evaluation_case()
                target = reference if field == 'timeline_review_complete' else reference['events'][0]
                target[field] = 'false'
                with self.assertRaises(ValueError):
                    evaluate_swing_events(model, reference)

    def test_valid_legacy_aliases_without_primary_fields_remain_supported(self):
        model, reference = evaluation_case()
        reference['events'][0].pop('frames')
        reference['events'][0].update(start_frame=100, contact_frame=110, end_frame=120)
        report = evaluate_swing_events(model, reference)
        self.assertEqual(report['events'][0]['contact_delta_frames'], 0)
        self.assertEqual(report['events'][0]['event_iou'], 1.)

    def test_writer_cannot_replace_inputs_through_direct_symlink_or_hardlink_output(self):
        model, reference = evaluation_case()
        for source_name in ('events.json', 'annotations.json'):
            for kind in ('direct', 'symlink', 'hardlink'):
                with self.subTest(source=source_name, kind=kind):
                    with tempfile.TemporaryDirectory() as tmp:
                        directory = Path(tmp)
                        events, annotations = directory / 'events.json', directory / 'annotations.json'
                        events.write_text(json.dumps(model))
                        annotations.write_text(json.dumps(reference))
                        original = (events.read_bytes(), annotations.read_bytes())
                        source = directory / source_name
                        output = source if kind == 'direct' else directory / f'{source.name}.{kind}'
                        if kind == 'symlink':
                            output.symlink_to(source)
                        elif kind == 'hardlink':
                            os.link(source, output)
                        with self.assertRaises(ValueError):
                            write_evaluation_report(str(events), str(annotations), str(output))
                        self.assertEqual((events.read_bytes(), annotations.read_bytes()), original)

    def test_valid_cli_writes_qualified_comparison_without_mutating_inputs(self):
        model, reference = evaluation_case()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            events, annotations, output = (directory / name for name in
                                           ('events.json', 'annotations.json', 'evaluation.json'))
            events.write_text(json.dumps(model))
            annotations.write_text(json.dumps(reference))
            original = (events.read_bytes(), annotations.read_bytes())
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('swing_evaluation.py')),
                                     '--events', str(events), '--annotations', str(annotations),
                                     '--output', str(output)], capture_output=True, text=True, timeout=8)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report['events'][0]['contact_delta_frames'], 0)
            self.assertEqual(report['settings']['identity_policy'], 'evaluation_identity_v1_strict_source_frames')
            self.assertIsNone(report['summary']['contact_accuracy'])
            self.assertFalse(report['reference_provenance']['independence_verified'])
            self.assertEqual((events.read_bytes(), annotations.read_bytes()), original)

    def test_cli_rejects_invalid_identity_without_creating_output(self):
        model, reference = evaluation_case()
        reference['events'][0]['frames']['contact'] = True
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            events = directory / 'events.json'
            annotations = directory / 'annotations.json'
            output = directory / 'evaluation.json'
            events.write_text(json.dumps(model))
            annotations.write_text(json.dumps(reference))
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('swing_evaluation.py')),
                                     '--events', str(events), '--annotations', str(annotations),
                                     '--output', str(output)], capture_output=True, text=True, timeout=8)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('非负安全整数', result.stderr)
            self.assertFalse(output.exists())

    def test_valid_inputs_keep_matching_statistics_and_are_not_mutated(self):
        model, reference = evaluation_case()
        original = deepcopy((model, reference))
        report = evaluate_swing_events(model, reference)
        self.assertEqual(report['summary']['true_positive_count'], 1)
        self.assertEqual(report['summary']['false_positive_count'], 0)
        self.assertEqual(report['summary']['precision'], 1.)
        self.assertEqual(report['events'][0]['event_iou'], 1.)
        self.assertEqual(report['events'][0]['contact_delta_frames'], 0)
        self.assertFalse(report['reference_provenance']['accuracy_validated'])
        self.assertEqual((model, reference), original)


if __name__ == '__main__':
    unittest.main()
