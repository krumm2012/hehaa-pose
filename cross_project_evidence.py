"""Associate immutable Analyzer and Vision evidence using exact input identity.

Different encodings require a separate source-frame mapping. Filenames, average
FPS and model agreement never establish that mapping.
"""
import json
import hashlib
import math
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

import numpy as np

from session_evidence_bundle import load_evidence_manifest, sha256_file, verify_evidence_manifest


def local_viewer_url(value):
    if not isinstance(value, str):
        raise ValueError('viewer URL must be a string')
    parsed = urlsplit(value)
    if (parsed.scheme not in ('http', 'https') or parsed.hostname not in ('127.0.0.1', 'localhost', '::1')
            or parsed.username is not None or parsed.password is not None
            or (parsed.port is not None and parsed.port <= 0)):
        raise ValueError('a credential-free loopback viewer URL is required')
    return value


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def verify_viewer_metadata(viewer_url, source_sha256, frames, timeout=3):
    url = urljoin(local_viewer_url(viewer_url), 'mesh_meta.json')
    try:
        with build_opener(_NoRedirect).open(url, timeout=timeout) as response:
            payload = response.read(4 * 1024 * 1024 + 1)
        if len(payload) > 4 * 1024 * 1024:
            raise ValueError('Viewer metadata exceeds the review size limit')
        meta = json.loads(payload)
        matched = (isinstance(meta, dict) and meta.get('video_sha256') == source_sha256
                   and type(meta.get('frames')) is int and meta['frames'] == frames)
        return {'status': 'verified' if matched else 'mismatched', 'metadata_url': url,
                'mesh_meta_sha256': hashlib.sha256(payload).hexdigest(),
                'reason': None if matched else 'Viewer belongs to a different source or frame sequence'}
    except (OSError, ValueError) as exc:
        return {'status': 'unavailable', 'metadata_url': url, 'reason': str(exc)}


def associate_evidence(manifest_path, vision_result, vision_source, reconstruction, viewer_url):
    manifest_path = Path(manifest_path).resolve()
    vision_result, vision_source, reconstruction = map(Path, (vision_result, vision_source, reconstruction))
    viewer_url = local_viewer_url(viewer_url)
    manifest = load_evidence_manifest(str(manifest_path))
    verification = verify_evidence_manifest(str(manifest_path))
    roles = {}
    for role in ('source_video', 'frame_journal', 'event_snapshot'):
        entries = [x for x in manifest['artifacts'] if x['role'] == role]
        verified = [x for x in verification['artifacts'] if x['role'] == role]
        if len(entries) != 1 or len(verified) != 1 or not verified[0]['valid']:
            raise ValueError('missing, ambiguous or changed Analyzer artifact: ' + role)
        path = Path(entries[0]['path'])
        roles[role] = (manifest_path.parent / path).resolve() if not path.is_absolute() else path
    frames = {}
    for line in roles['frame_journal'].read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        fid = row.get('frame_id')
        if type(fid) is not int or fid < 0 or fid in frames:
            raise ValueError('invalid or duplicate Analyzer frame identity')
        frames[fid] = row
    if not frames:
        raise ValueError('Analyzer source frames required')
    event_document = json.loads(roles['event_snapshot'].read_text(encoding='utf-8'))
    events = event_document.get('events') if isinstance(event_document, dict) else event_document
    if not isinstance(events, list):
        raise ValueError('Analyzer events must be a list')
    meta_path, multiview_path = vision_result / 'mesh_meta.json', vision_result / 'multiview_manifest.json'
    meta, multiview = (json.loads(path.read_text(encoding='utf-8')) for path in (meta_path, multiview_path))
    source_hash = sha256_file(vision_source)
    with np.load(reconstruction, allow_pickle=False) as native:
        native_hash = str(native['video_sha256'].item())
        native_frame_count = len(native['joints'])
    if source_hash != meta.get('video_sha256') or source_hash != multiview.get('source_sha256') or source_hash != native_hash:
        raise ValueError('Vision source, reconstruction and metadata hashes do not match')
    count = meta.get('frames')
    if (type(count) is not int or count <= 0 or type(multiview.get('frames')) is not int
            or multiview['frames'] != count or native_frame_count != count):
        raise ValueError('Vision source frame counts do not match')
    tracking = multiview.get('tracking')
    tracking_verified = (isinstance(tracking, list) and len(tracking) == count
                         and all(isinstance(row, dict) and type(row.get('frame')) is int
                                 and row['frame'] == i for i, row in enumerate(tracking)))
    analyzer_hash = sha256_file(roles['source_video'])
    same_source = analyzer_hash == source_hash
    source_frames_verified = all(
        isinstance(row.get('source_time'), dict)
        and row['source_time'].get('schema_version') == 'tennis.source-time.v1'
        and row['source_time'].get('source_kind') == 'video_file'
        and type(row['source_time'].get('source_frame_id')) is int
        and row['source_time']['source_frame_id'] == fid
        for fid, row in frames.items())
    in_range = all(fid < count for fid in frames)
    blockers = []
    if not same_source:
        blockers.append('different_input_video_hashes_need_source_frame_mapping')
    if not source_frames_verified:
        blockers.append('Analyzer_source_frame_contract_unverified')
    if not tracking_verified:
        blockers.append('Vision_decoded_frame_index_contract_unverified')
    if same_source and not in_range:
        blockers.append('Analyzer_frame_outside_Vision_source')
    ready = not blockers
    linked_events = []
    for event in events:
        if (type(event.get('start_frame')) is not int or type(event.get('end_frame')) is not int
                or event['start_frame'] > event['end_frame']):
            raise ValueError('invalid event boundaries')
        anchors = {}
        for field in ('start_frame', 'contact_frame', 'peak_frame', 'end_frame'):
            fid = event.get(field)
            if type(fid) is not int or fid < 0:
                raise ValueError('invalid event frame identity: ' + field)
            row = frames.get(fid)
            value = (row or {}).get('source_time')
            source = value if isinstance(value, dict) else {}
            seconds = source.get('timestamp_seconds')
            valid_seconds = (type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0)
            anchors[field] = {
                'Analyzer_frame_id': fid,
                'source_timestamp_seconds': seconds if valid_seconds else None,
                'source_time_quality': source.get('quality'),
                'Vision_frame_index': fid if ready and row is not None else None,
                'status': 'linked_same_input_frame' if ready and row is not None else
                          ('missing_Analyzer_frame_evidence' if ready else 'pending_source_frame_mapping'),
            }
        linked_events.append({'event_id': event.get('event_id'), 'stroke_type': event.get('stroke_type'),
                              'anchors': anchors, 'viewer_url': viewer_url})
    return {
        'schema': 'tennis.cross-project-evidence.v1',
        'status': 'linked_same_input_video' if ready else 'pending_source_frame_mapping',
        'same_input_video': same_source, 'source_frame_ids_verified': ready, 'blockers': blockers,
        'Analyzer': {'source_sha256': analyzer_hash, 'recorded_frames': len(frames),
                     'manifest_sha256': sha256_file(manifest_path), 'bundle_id': manifest.get('bundle_id'),
                     'frame_journal_sha256': sha256_file(roles['frame_journal']),
                     'event_snapshot_sha256': sha256_file(roles['event_snapshot'])},
        'Vision': {'source_sha256': source_hash, 'frames': count, 'viewer_url': viewer_url,
                   'mesh_meta_sha256': sha256_file(meta_path), 'multiview_manifest_sha256': sha256_file(multiview_path),
                   'reconstruction_sha256': sha256_file(reconstruction), 'tracking_indices_verified': tracking_verified},
        'events': linked_events, 'accuracy_validated': False, 'coach_eligible': False,
        'frame_identity_semantics': 'same input bytes plus explicit zero-based source and exporter tracking indices',
        'limitations': ['model_reconstruction_is_not_independent_ground_truth',
                       'same_input_frame_is_not_verified_sensor_exposure',
                       'derived_video_relationship_to_original_is_not_established_by_this_tool',
                       'body_identity_and_anatomical_joint_correspondence_require_separate_validation'],
    }
