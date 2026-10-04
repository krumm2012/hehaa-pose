"""Exact source identity and explicit frame indices precede cross-project links."""
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from cross_project_evidence import associate_evidence, local_viewer_url, verify_viewer_metadata
from session_evidence_bundle import sha256_file


class CrossProjectEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source.mp4'
        self.source.write_bytes(b'synthetic source identity; contract tests only')
        self.frame_path, self.event_path = self.root / 'frames.jsonl', self.root / 'events.json'
        self.frames = [{'frame_id': i, 'source_time': {
            'schema_version': 'tennis.source-time.v1', 'source_kind': 'video_file',
            'source_frame_id': i, 'timestamp_seconds': seconds, 'quality': 'reported'}}
            for i, seconds in enumerate((0.0, .021, .103))]
        self.events = {'events': [{'event_id': 1, 'stroke_type': 'Forehand',
                   'start_frame': 0, 'contact_frame': 1, 'peak_frame': 1, 'end_frame': 2}]}
        self.manifest_path = self.root / 'manifest.json'
        self.write_analyzer()
        self.result = self.root / 'result'
        self.result.mkdir()
        digest = sha256_file(self.source)
        self.meta = {'video_sha256': digest, 'frames': 3}
        self.multiview = {'source_sha256': digest, 'frames': 3, 'tracking': [{'frame': i} for i in range(3)]}
        self.reconstruction = self.root / 'reconstruction.npz'
        self.write_vision()

    def write_analyzer(self):
        self.frame_path.write_text(''.join(json.dumps(x) + '\n' for x in self.frames))
        self.event_path.write_text(json.dumps(self.events))
        entries = [{'role': role, 'path': path.name, 'bytes': path.stat().st_size,
                    'sha256': sha256_file(path), 'required_for_replay': True}
                   for role, path in [('source_video', self.source), ('frame_journal', self.frame_path),
                                      ('event_snapshot', self.event_path)]]
        self.manifest_path.write_text(json.dumps({'schema_version': 'tennis.replay-evidence-bundle.v1',
                                     'bundle_id': 'synthetic', 'artifacts': entries}))

    def write_vision(self, source=None):
        digest = sha256_file(source or self.source)
        self.meta['video_sha256'] = digest
        self.multiview['source_sha256'] = digest
        (self.result / 'mesh_meta.json').write_text(json.dumps(self.meta))
        (self.result / 'multiview_manifest.json').write_text(json.dumps(self.multiview))
        np.savez(self.reconstruction, video_sha256=np.array(digest), joints=np.zeros((self.meta['frames'], 11, 3)))

    def associate(self, source=None):
        return associate_evidence(self.manifest_path, self.result, source or self.source,
                                  self.reconstruction, 'http://127.0.0.1:18769/datasets/example/result/viewer.html')

    def test_same_bytes_preserve_source_frames_and_nonuniform_times_without_truth_claim(self):
        result = self.associate()
        self.assertEqual(result['status'], 'linked_same_input_video')
        self.assertTrue(result['source_frame_ids_verified'])
        anchors = result['events'][0]['anchors']
        self.assertEqual(anchors['start_frame']['Vision_frame_index'], 0)
        self.assertEqual(anchors['start_frame']['source_timestamp_seconds'], 0)
        self.assertEqual(anchors['contact_frame']['source_timestamp_seconds'], .021)
        self.assertFalse(result['accuracy_validated'])
        self.assertFalse(result['coach_eligible'])

    def test_different_encoding_or_filename_cannot_be_joined_by_fps_or_frame_number(self):
        other = self.root / 'same_name_different_encoding.mp4'
        other.write_bytes(b'different normalized video')
        self.write_vision(other)
        result = self.associate(other)
        self.assertEqual(result['status'], 'pending_source_frame_mapping')
        self.assertFalse(result['source_frame_ids_verified'])
        self.assertIn('different_input_video_hashes_need_source_frame_mapping', result['blockers'])
        self.assertIsNone(result['events'][0]['anchors']['contact_frame']['Vision_frame_index'])

    def test_changed_or_ambiguous_analyzer_artifacts_are_rejected(self):
        self.source.write_bytes(b'changed source')
        with self.assertRaisesRegex(ValueError, 'changed Analyzer'): self.associate()
        self.write_analyzer()
        manifest = json.loads(self.manifest_path.read_text())
        manifest['artifacts'].append(dict(manifest['artifacts'][0]))
        self.manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'ambiguous'): self.associate()

    def test_vision_hash_or_native_frame_count_mismatch_is_rejected(self):
        self.meta['video_sha256'] = 'a' * 64
        (self.result / 'mesh_meta.json').write_text(json.dumps(self.meta))
        with self.assertRaisesRegex(ValueError, 'hashes'): self.associate()
        self.write_vision()
        np.savez(self.reconstruction, video_sha256=np.array(sha256_file(self.source)), joints=np.zeros((2, 11, 3)))
        with self.assertRaisesRegex(ValueError, 'counts'): self.associate()

    def test_legacy_or_mismatched_source_ids_and_tracking_cannot_supply_links(self):
        self.frames[0]['source_time'] = None
        self.write_analyzer()
        self.assertIn('Analyzer_source_frame_contract_unverified', self.associate()['blockers'])
        self.frames[0]['source_time'] = self.frames[1]['source_time']
        self.write_analyzer()
        self.assertFalse(self.associate()['source_frame_ids_verified'])
        self.multiview['tracking'][1]['frame'] = 0
        self.write_vision()
        self.assertIn('Vision_decoded_frame_index_contract_unverified', self.associate()['blockers'])

    def test_duplicate_and_invalid_event_frames_are_rejected(self):
        self.frames.append(dict(self.frames[0]))
        self.write_analyzer()
        with self.assertRaisesRegex(ValueError, 'duplicate'): self.associate()
        self.frames.pop()
        self.events['events'][0]['start_frame'] = 3
        self.write_analyzer()
        with self.assertRaisesRegex(ValueError, 'boundaries'): self.associate()

    def test_missing_anchor_evidence_stays_missing_and_invalid_time_is_null(self):
        self.frames.pop(1)
        self.frames[0]['source_time']['timestamp_seconds'] = float('nan')
        self.write_analyzer()
        result = self.associate()
        anchors = result['events'][0]['anchors']
        self.assertIsNone(anchors['start_frame']['source_timestamp_seconds'])
        self.assertIsNone(anchors['contact_frame']['Vision_frame_index'])
        self.assertEqual(anchors['contact_frame']['status'], 'missing_Analyzer_frame_evidence')
        json.dumps(result, allow_nan=False)

    def test_viewer_urls_are_local_and_credential_free(self):
        for value in ('javascript:alert(1)', 'https://example.com/', 'http://user:secret@localhost/',
                      'http://127.0.0.1:99999/', 'http://127.0.0.1:0/'):
            with self.subTest(value=value), self.assertRaises(ValueError): local_viewer_url(value)
        self.assertEqual(local_viewer_url('http://localhost:18769/'), 'http://localhost:18769/')

    def test_alternate_pixel_source_requires_matching_geometry_without_rewriting_reference(self):
        import cv2
        from scripts.run_pixel_acceptance import select_source
        original = self.source.read_bytes()
        alternate = self.root / 'alternate.avi'
        writer = cv2.VideoWriter(str(alternate), cv2.VideoWriter_fourcc(*'MJPG'), 25, (100, 80))
        self.assertTrue(writer.isOpened())
        try:
            writer.write(np.zeros((80, 100, 3), dtype=np.uint8))
        finally:
            writer.release()
        entry = {'path': self.source.name, 'sha256': sha256_file(self.source)}
        self.assertEqual(select_source(self.manifest_path, entry, {'frame_size': [100, 80]}), self.source.resolve())
        self.assertEqual(select_source(self.manifest_path, entry, {'frame_size': [100, 80]}, alternate), alternate.resolve())
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            select_source(self.manifest_path, entry, {'frame_size': [200, 160]}, alternate)
        self.assertEqual(self.source.read_bytes(), original)
        self.source.write_bytes(b'changed after the frozen manifest was written')
        with self.assertRaisesRegex(ValueError, 'Frozen source video'):
            select_source(self.manifest_path, entry, {'frame_size': [100, 80]})

    def test_viewer_endpoint_is_bound_to_source_and_frame_count(self):
        payload = {'video_sha256': sha256_file(self.source), 'frames': 3}
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f'http://127.0.0.1:{server.server_port}/result/viewer.html'
            self.assertEqual(verify_viewer_metadata(url, sha256_file(self.source), 3)['status'], 'verified')
            payload['video_sha256'] = 'b' * 64
            self.assertEqual(verify_viewer_metadata(url, sha256_file(self.source), 3)['status'], 'mismatched')
            payload['video_sha256'] = sha256_file(self.source)
            payload['frames'] = 4
            self.assertEqual(verify_viewer_metadata(url, sha256_file(self.source), 3)['status'], 'mismatched')
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_pixel_cli_rejects_changed_or_missing_optional_source_before_creating_output(self):
        from scripts.run_pixel_acceptance import main
        manifest = json.loads(self.manifest_path.read_text())
        manifest['capture'] = {'roi': {'frame_size': [100, 80],
                              'points': [[0, 0], [99, 0], [99, 79], [0, 79]],
                              'mirror_view': [50, 0, 50, 80], 'front_view': [0, 0, 50, 80]}}
        manifest['artifacts'][0]['required_for_replay'] = False
        self.manifest_path.write_text(json.dumps(manifest))
        output = self.root / 'rejected_pixel_run'
        argv = ['run_pixel_acceptance.py', '--manifest', str(self.manifest_path),
                '--output', str(output)]
        self.source.write_bytes(b'changed after freezing; FrameRecord replay remains valid')
        for source_state in ('changed', 'missing'):
            if source_state == 'missing':
                self.source.unlink()
            with self.subTest(source_state=source_state), patch('sys.argv', argv), \
                    patch('scripts.run_pixel_acceptance.subprocess.Popen',
                          side_effect=AssertionError('rejected input must not start inference')):
                with self.assertRaisesRegex(ValueError, 'Frozen source video'):
                    main()
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
