"""Independent annotators must resolve disagreements without averaging truth."""
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from joint_label_adjudication import build_adjudication, finalize_adjudication, document_digest


def reference(annotator):
    return {'schema': 'tennis.independent-joint-labels.v1', 'source_sha256': 'a' * 64,
            'annotator_id': annotator, 'confirmed': True,
            'coordinate_space': 'original_source_pixels', 'frame_index_base': 0,
            'requested_joints': ['left_shoulder'],
            'frames': [{'frame_id': 175, 'width': 100, 'height': 80, 'file': 'frame_175.png'}],
            'labels': {'175:front:left_shoulder': {'visible': True, 'x': 0, 'y': 20},
                       '175:back:left_shoulder': {'visible': False, 'x': None, 'y': None,
                                                  'reason': 'not_identifiable'}}}


def decisions(plan, labels=None, confirmed=True):
    return {'schema': 'tennis.joint-adjudication-decisions.v1',
            'plan_sha256': document_digest(plan), 'annotator_id': 'arbiter',
            'confirmed': confirmed, 'labels': labels or {}}


class JointLabelAdjudicationTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node required for actual board script validation')
    def test_rendered_page_script_parses_with_user_text_and_template_markers(self):
        from scripts.adjudicate_joint_labels import render_html
        left, right = reference('a'), reference('b')
        left['labels']['175:front:left_shoulder']['reason'] = 'PLAN_HASH </script> 原帧复核'
        page = render_html(build_adjudication(left, right, 10))
        script = page.split('<script>', 1)[1].split('</script>', 1)[0]
        result = subprocess.run(['node', '--check'], input=script, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('\\u003c/script>', script)

    def test_agreement_keeps_zero_visibility_provenance_and_originals(self):
        left, right = reference('a'), reference('b')
        before = copy.deepcopy((left, right))
        plan = build_adjudication(left, right, tolerance_px=10)
        self.assertEqual(plan['summary']['agreement_count'], 2)
        self.assertEqual(plan['summary']['review_required_count'], 0)
        result = finalize_adjudication(left, right, plan, decisions(plan))
        self.assertEqual(result['labels']['175:front:left_shoulder']['x'], 0)
        self.assertFalse(result['labels']['175:back:left_shoulder']['visible'])
        self.assertEqual(result['requested_joints'], ['left_shoulder'])
        self.assertEqual(result['adjudication']['references'][0]['document_sha256'], document_digest(left))
        self.assertFalse(result['accuracy_validated'])
        self.assertEqual((left, right), before)

    def test_small_coordinate_disagreement_requires_explicit_decision(self):
        left, right = reference('a'), reference('b')
        right['labels']['175:front:left_shoulder']['x'] = 2
        plan = build_adjudication(left, right, tolerance_px=10)
        row = next(r for r in plan['rows'] if r['key'] == '175:front:left_shoulder')
        self.assertEqual(row['distance_px'], 2)
        self.assertTrue(row['within_reporting_tolerance'])
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            finalize_adjudication(left, right, plan, decisions(plan))
        result = finalize_adjudication(left, right, plan, decisions(plan, {
            row['key']: {'choice': 'right', 'reason': 'checked original source frame'}}))
        self.assertEqual(result['labels'][row['key']]['x'], 2)
        self.assertNotEqual(result['labels'][row['key']]['x'], 1)

    def test_visibility_missing_and_custom_points_need_source_review(self):
        left, right = reference('a'), reference('b')
        del left['labels']['175:back:left_shoulder']
        del right['labels']['175:back:left_shoulder']
        right['labels']['175:front:left_shoulder'] = {'visible': False, 'x': None, 'y': None}
        plan = build_adjudication(left, right, tolerance_px=5)
        self.assertEqual(plan['summary']['review_required_count'], 2)
        resolved = {'175:front:left_shoulder': {'choice': 'right', 'reason': 'occluded in source'},
                    '175:back:left_shoulder': {'choice': 'custom', 'reason': 'independently checked source',
                        'point': {'visible': True, 'x': 99, 'y': 79}}}
        result = finalize_adjudication(left, right, plan, decisions(plan, resolved))
        self.assertFalse(result['labels']['175:front:left_shoulder']['visible'])
        self.assertEqual(len(result['labels']), 2)
        resolved['175:back:left_shoulder']['point']['x'] = 100
        with self.assertRaisesRegex(ValueError, 'outside'):
            finalize_adjudication(left, right, plan, decisions(plan, resolved))

    def test_assisted_or_model_origin_cannot_become_independent_consensus(self):
        for change in ({'schema': 'tennis.assisted-joint-review.v1'},
                       {'annotation_mode': 'model_assisted'}, {'independent_reference': False},
                       {'independent_reference': 1}, {'independent_reference': 'true'},
                       {'model_suggestions': {'175:front:left_shoulder': {'x': 0, 'y': 20}}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_adjudication(reference('a'), {**reference('b'), **change}, 10)
        assisted = reference('b')
        assisted['labels']['175:front:left_shoulder']['origin'] = 'human_bulk_accepted_model'
        with self.assertRaises(ValueError):
            build_adjudication(reference('a'), assisted, 10)

    def test_changed_input_or_plan_cannot_reuse_confirmation(self):
        left, right = reference('a'), reference('b')
        plan = build_adjudication(left, right, 10)
        confirm = decisions(plan)
        left['labels']['175:front:left_shoulder']['x'] = 1
        with self.assertRaisesRegex(ValueError, 'changed'):
            finalize_adjudication(left, right, plan, confirm)
        left = reference('a')
        changed_plan = copy.deepcopy(plan)
        changed_plan['rows'][0]['status'] = 'agreement'
        with self.assertRaisesRegex(ValueError, 'changed'):
            finalize_adjudication(left, right, changed_plan, confirm)

    def test_frame_joint_and_coordinate_contracts_must_match(self):
        for mutate in ('source', 'frame', 'joints', 'duplicate', 'boolean_coordinate'):
            right = reference('b')
            if mutate == 'source': right['source_sha256'] = 'b' * 64
            if mutate == 'frame': right['frames'][0]['width'] = 101
            if mutate == 'joints': right['requested_joints'] = ['right_shoulder']
            if mutate == 'duplicate': right['frames'].append(dict(right['frames'][0]))
            if mutate == 'boolean_coordinate': right['labels']['175:front:left_shoulder']['x'] = False
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                build_adjudication(reference('a'), right, 10)

    def test_drafts_same_annotator_and_unconfirmed_decisions_cannot_finalize(self):
        left, right = reference('a'), reference('b')
        right['confirmed'] = False
        plan = build_adjudication(left, right, 10)
        self.assertEqual(plan['status'], 'pending_reference_confirmation')
        with self.assertRaises(ValueError): finalize_adjudication(left, right, plan, decisions(plan))
        right = reference('a')
        with self.assertRaises(ValueError): build_adjudication(left, right, 10)
        right = reference('b'); plan = build_adjudication(left, right, 10)
        with self.assertRaises(ValueError):
            finalize_adjudication(left, right, plan, decisions(plan, confirmed=False))

    def test_cli_preserves_sources_refuses_overwrite_and_feeds_independent_evaluator(self):
        import cv2
        import numpy as np
        from joint_annotation_evaluation import evaluate_joint_labels
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'synthetic.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 25, (100, 80))
            self.assertTrue(writer.isOpened(), 'synthetic video writer must be available')
            try:
                writer.write(np.zeros((80, 100, 3), dtype=np.uint8))
            finally:
                writer.release()
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            left, right = reference('a'), reference('b')
            for document in (left, right):
                document['source_sha256'] = digest
                document['frames'][0]['frame_id'] = 0
                document['labels'] = {key.replace('175:', '0:'): p for key, p in document['labels'].items()}
            right['labels']['0:front:left_shoulder']['x'] = 2
            left_path, right_path = root / 'left.json', root / 'right.json'
            for path, document in ((left_path, left), (right_path, right)):
                path.write_text(json.dumps(document), encoding='utf-8')
            originals = {path: path.read_bytes() for path in (source, left_path, right_path)}
            script = Path(__file__).parent / 'scripts/adjudicate_joint_labels.py'

            def run(*arguments):
                return subprocess.run([sys.executable, str(script), *map(str, arguments)],
                                      text=True, capture_output=True)

            board = root / 'board'
            arguments = ('create', '--left', left_path, '--right', right_path, '--source', source,
                         '--output', board, '--tolerance-px', '10')
            created = run(*arguments)
            self.assertEqual(created.returncode, 0, created.stderr)
            plan = json.loads((board / 'plan.json').read_text())
            self.assertEqual(plan['summary']['review_required_count'], 1)
            pixels = cv2.imread(str(board / 'frame_0.png'))
            self.assertEqual(pixels.shape[:2], (80, 100))
            self.assertEqual(json.loads((board / 'reference_left.json').read_text()), left)
            self.assertEqual(json.loads((board / 'reference_right.json').read_text()), right)
            board_before = {path.name: path.read_bytes() for path in board.iterdir()}
            self.assertNotEqual(run(*arguments).returncode, 0)
            self.assertEqual({path.name: path.read_bytes() for path in board.iterdir()}, board_before)

            choice_path = root / 'decisions.json'
            output = root / 'final.json'
            choice_path.write_text(json.dumps(decisions(plan)), encoding='utf-8')
            finish = ('finalize', '--left', left_path, '--right', right_path, '--plan', board / 'plan.json',
                      '--decisions', choice_path, '--output', output)
            self.assertNotEqual(run(*finish).returncode, 0)
            self.assertFalse(output.exists(), 'unresolved decisions cannot publish a final reference')
            choice_path.write_text(json.dumps(decisions(plan, {
                '0:front:left_shoulder': {'choice': 'right', 'reason': 'checked synthetic source frame'}})),
                encoding='utf-8')
            final = run(*finish)
            self.assertEqual(final.returncode, 0, final.stderr)
            labels = json.loads(output.read_text())
            predictions = {'schema': 'tennis.pose-resolution-audit.v1', 'source_sha256': digest,
                           'scales': [1.0], 'samples': {'0': {'1.0': {'front': {'left_shoulder': {
                               'x': 2, 'y': 20, 'observed': True, 'confidence': .9, 'source_frame_id': 0}}}}}}
            evaluated = evaluate_joint_labels(labels, predictions, tolerance_px=10)
            self.assertEqual(evaluated['status'], 'evaluated_independent_labels')
            self.assertEqual(evaluated['planned_labels'], 2)
            group = next(g for g in evaluated['groups'] if g['view'] == 'front' and g['joint'] == 'all')
            self.assertEqual(group['mean_error_px'], 0)
            self.assertEqual(group['qualified_output_rate'], 1)
            before_final = output.read_bytes()
            self.assertNotEqual(run(*finish).returncode, 0)
            self.assertEqual(output.read_bytes(), before_final)
            self.assertEqual({path: path.read_bytes() for path in originals}, originals)

            other_source = root / 'changed.avi'
            other_source.write_bytes(source.read_bytes() + b'changed')
            rejected = root / 'rejected'
            mismatch = run('create', '--left', left_path, '--right', right_path, '--source', other_source,
                           '--output', rejected, '--tolerance-px', '10')
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertFalse(rejected.exists(), 'source mismatch must fail before creating an output directory')


if __name__ == '__main__':
    unittest.main()
