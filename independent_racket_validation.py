"""Held-out 2D point/contact errors with explicit human-reference provenance."""
import math
from statistics import mean, median
from joint_racket_2d_analysis import RACKET_POINTS, finite


def evaluate(prediction, reference, excluded_source_hashes, tolerance_px):
    if not finite(tolerance_px) or tolerance_px <= 0:
        raise ValueError('Explicit positive point tolerance in pixels required')
    source_hash = reference.get('source_sha256')
    if (not isinstance(source_hash, str) or len(source_hash) != 64
            or any(c not in '0123456789abcdef' for c in source_hash)):
        raise ValueError('Source SHA-256 required')
    if prediction.get('source_sha256') != source_hash:
        raise ValueError('Prediction/reference source mismatch')
    if source_hash in set(excluded_source_hashes):
        raise ValueError('Source belongs to development/acceptance material, not held-out')
    if reference.get('schema') != 'tennis.independent-racket-reference.v1':
        raise ValueError('Unsupported reference')
    if prediction.get('schema') != 'tennis.independent-racket-prediction.v1':
        raise ValueError('Unsupported prediction')
    for document in (prediction, reference):
        if document.get('coordinate_space') != 'original_source_pixels' or document.get('frame_index_base') != 0:
            raise ValueError('Original source pixel/frame coordinates required')
    for key in ('reviewer', 'split_provenance', 'reference_evidence'):
        if not isinstance(reference.get(key), str) or not reference[key].strip():
            raise ValueError('Missing reference provenance: ' + key)
    if reference.get('model_assisted') is not False or reference.get('confirmed') is not True:
        raise ValueError('Requires explicitly confirmed reference annotated without model overlays')
    if not prediction.get('model_sha256'):
        raise ValueError('Frozen prediction model identity required')
    size = reference.get('image_size')
    if (not isinstance(size, list) or len(size) != 2
            or any(type(v) is not int or v <= 0 for v in size) or prediction.get('image_size') != size):
        raise ValueError('Matching image dimensions required')
    def index(rows):
        result = {}
        for row in rows:
            fid = row.get('frame_id')
            if type(fid) is not int or fid < 0 or fid in result:
                raise ValueError('Duplicate/invalid frame identity')
            result[fid] = row
        return result
    actual = index(reference.get('frames') or [])
    predicted = index(prediction.get('frames') or [])
    if not actual: raise ValueError('No reviewed reference frames')
    def point(value):
        if (not isinstance(value, list) or len(value) != 2 or not all(finite(x) for x in value)
                or not 0 <= value[0] < size[0] or not 0 <= value[1] < size[1]):
            raise ValueError('Invalid source pixel point')
        return value
    groups = []
    for name in RACKET_POINTS:
        errors = []; visible = 0; missing = 0
        for fid, row in actual.items():
            target = (row.get('points') or {}).get(name)
            if target is None: continue  # Human marked not visible; not an inferred coordinate.
            visible += 1; target = point(target)
            observed = (predicted.get(fid, {}).get('points') or {}).get(name)
            if observed is None:
                missing += 1; continue
            errors.append(math.dist(point(observed), target))
        groups.append({'point': name, 'visible_reference_count': visible, 'matched_count': len(errors),
                       'missing_predictions': missing, 'mean_error_px': mean(errors) if errors else None,
                       'median_error_px': median(errors) if errors else None,
                       'max_error_px': max(errors) if errors else None,
                       'within_tolerance_over_visible': sum(e <= tolerance_px for e in errors)/visible if visible else None})
    # Manual independent intervals are paired using explicit review correspondences.
    # Pairing does not measure event detector recall/precision.
    contacts = []
    seen = set()
    for item in reference.get('contact_pairs') or []:
        identity = item.get('reference_id')
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError('Duplicate/invalid reference contact identity')
        seen.add(identity)
        start, end = item.get('start_frame'), item.get('end_frame')
        candidate = (prediction.get('contacts') or {}).get(identity)
        if type(start) is not int or type(end) is not int or start < 0 or end < start:
            raise ValueError('Invalid reviewed contact interval')
        for fid in (start, end):
            if fid not in actual or not finite(actual[fid].get('source_seconds')):
                raise ValueError('Reference interval requires bound source PTS')
        if actual[end]['source_seconds'] < actual[start]['source_seconds']:
            raise ValueError('Invalid reference PTS order')
        error = error_ms = None
        if candidate is not None:
            if type(candidate) is not int or candidate not in actual:
                raise ValueError('Prediction requires independently bound source PTS')
            error = max(start-candidate, candidate-end, 0)
            t = actual[candidate].get('source_seconds')
            if not finite(t): raise ValueError('Unqualified candidate PTS')
            error_ms = 1000 * max(actual[start]['source_seconds']-t, t-actual[end]['source_seconds'], 0)
        contacts.append({'reference_id': identity, 'predicted_frame': candidate,
                         'reference_interval': [start, end], 'outside_interval_frames': error,
                         'outside_interval_ms': error_ms})
    return {'schema': 'tennis.independent-racket-errors.v1', 'source_sha256': source_hash,
            'point_tolerance_px': tolerance_px, 'tolerance_is_acceptance_standard': False,
            'points': groups, 'contact_pairs': contacts, 'reviewed_frame_count': len(actual),
            'event_detection_precision_recall_evaluated': False,
            'metric_3d_speed_accuracy_evaluated': False,
            'independence': 'human_declared_provenance_plus_known_hash_exclusion_not_training_data_audit',
            'active_or_exclusive_allowed': False, 'decision': 'errors_reported_requires_quality_review'}
