"""Compare independent human references and require explicit disagreement decisions.

The tolerance categorizes coordinate differences for review; it never permits
averaging points or certifies measurement/technique accuracy.
"""
import copy
import hashlib
import json
import math
import re

from joint_annotation_evaluation import JOINTS, finite

PLAN_SCHEMA = 'tennis.joint-adjudication-plan.v1'
DECISIONS_SCHEMA = 'tennis.joint-adjudication-decisions.v1'


def document_digest(document):
    """Hash the whole original JSON document, independent of serialization spacing."""
    payload = json.dumps(document, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _point(point, frame):
    if not isinstance(point, dict) or type(point.get('visible')) is not bool:
        raise ValueError('visibility must be explicit')
    if 'model' in str(point.get('origin', '')).lower() or any(k.startswith('model_') for k in point):
        raise ValueError('model-assisted points are not independent references')
    if point['visible']:
        if not all(finite(point.get(k)) for k in ('x', 'y')):
            raise ValueError('invalid independent point coordinates')
        if not (0 <= point['x'] < frame['width'] and 0 <= point['y'] < frame['height']):
            raise ValueError('point outside declared source frame')
    elif point.get('x') is not None or point.get('y') is not None:
        raise ValueError('unidentifiable points must not supply coordinates')
    return copy.deepcopy(point)


def _reference(document):
    if not isinstance(document, dict) or document.get('schema') != 'tennis.independent-joint-labels.v1':
        raise ValueError('independent joint-label schema required')
    if (document.get('annotation_mode') not in (None, 'independent', 'independent_adjudicated')
        or (document.get('independent_reference') is not None
            and document.get('independent_reference') is not True)
        or document.get('model_suggestions')):
        raise ValueError('model-assisted references cannot be adjudicated as independent')
    digest = document.get('source_sha256')
    if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('invalid source video hash')
    if (document.get('coordinate_space') != 'original_source_pixels'
        or type(document.get('frame_index_base')) is not int or document['frame_index_base'] != 0):
        raise ValueError('unsupported coordinate contract')
    requested = document.get('requested_joints', list(JOINTS))
    if (not isinstance(requested, list) or not requested or any(j not in JOINTS for j in requested)
        or len(set(requested)) != len(requested)):
        raise ValueError('invalid requested joint set')
    joints = tuple(j for j in JOINTS if j in requested)
    frames = {}
    declared_frames = document.get('frames')
    if not isinstance(declared_frames, list) or not declared_frames:
        raise ValueError('source frames required')
    for frame in declared_frames:
        if not isinstance(frame, dict): raise ValueError('invalid source frame')
        fid = frame.get('frame_id')
        if type(fid) is not int or fid < 0 or fid in frames:
            raise ValueError('invalid or duplicate source frame')
        if not all(type(frame.get(k)) is int and frame[k] > 0 for k in ('width', 'height')):
            raise ValueError('invalid frame dimensions')
        frames[fid] = {'frame_id': fid, 'width': frame['width'], 'height': frame['height']}
    labels = document.get('labels')
    if not isinstance(labels, dict): raise ValueError('labels must be an object')
    for key, point in labels.items():
        parts = key.split(':') if isinstance(key, str) else []
        if len(parts) != 3 or not parts[0].isdigit(): raise ValueError('invalid label key')
        fid, view, joint = int(parts[0]), parts[1], parts[2]
        if (key != f'{fid}:{view}:{joint}' or fid not in frames
            or view not in ('front', 'back') or joint not in joints):
            raise ValueError('label outside declared frame/view/joint set')
        _point(point, frames[fid])
    annotator = document.get('annotator_id')
    if annotator is not None and not isinstance(annotator, str):
        raise ValueError('annotator ID must be a string')
    return frames, joints, (annotator or '').strip()


def build_adjudication(left, right, tolerance_px):
    if not finite(tolerance_px) or tolerance_px <= 0:
        raise ValueError('reporting tolerance must be explicit, finite and positive')
    lf, lj, la = _reference(left)
    rf, rj, ra = _reference(right)
    if left['source_sha256'] != right['source_sha256'] or lf != rf or lj != rj:
        raise ValueError('reference source/frame/joint contracts must match')
    if la and ra and la == ra:
        raise ValueError('two different independent annotators required')
    ready = left.get('confirmed') is True and right.get('confirmed') is True and bool(la and ra)
    rows = []
    for fid in sorted(lf):
        for view in ('front', 'back'):
            for joint in lj:
                key = f'{fid}:{view}:{joint}'
                a, b = left['labels'].get(key), right['labels'].get(key)
                distance = None
                if a is None or b is None:
                    status = 'missing_annotation'
                elif a['visible'] != b['visible']:
                    status = 'visibility_disagreement'
                elif not a['visible']:
                    status = 'agree_unidentifiable'
                else:
                    distance = math.hypot(a['x'] - b['x'], a['y'] - b['y'])
                    status = 'agree_visible' if distance == 0 else 'position_disagreement'
                rows.append({'key': key, 'frame_id': fid, 'view': view, 'joint': joint,
                             'status': status, 'review_required': not status.startswith('agree_'),
                             'distance_px': distance,
                             'within_reporting_tolerance': distance <= tolerance_px if distance is not None else None,
                             'left': copy.deepcopy(a), 'right': copy.deepcopy(b)})
    agreement = sum(not row['review_required'] for row in rows)
    return {'schema': PLAN_SCHEMA, 'source_sha256': left['source_sha256'],
            'coordinate_space': 'original_source_pixels', 'frame_index_base': 0,
            'requested_joints': list(lj), 'frames': [lf[k] for k in sorted(lf)],
            'references': [{'side': side, 'annotator_id': annotator or None,
                            'confirmed': doc.get('confirmed') is True,
                            'document_sha256': document_digest(doc)}
                           for side, annotator, doc in [('left', la, left), ('right', ra, right)]],
            'digest_semantics': 'SHA256 of canonical JSON of each entire original document',
            'reporting_tolerance_px': tolerance_px,
            'tolerance_meaning': 'review grouping only; all unequal positions require human adjudication',
            'status': ('needs_adjudication' if agreement < len(rows) else 'ready_for_confirmation')
                      if ready else 'pending_reference_confirmation',
            'summary': {'planned_labels': len(rows), 'agreement_count': agreement,
                        'review_required_count': len(rows) - agreement,
                        'paired_count': sum(row['left'] is not None and row['right'] is not None for row in rows)},
            'rows': rows, 'accuracy_validated': False,
            'limitations': ['independence_is_declared_by_humans_not_proven_by_this_tool',
                            'agreement_does_not_prove_correct_coordinates',
                            'no_3d_kinematic_or_technique_accuracy_claim']}


def finalize_adjudication(left, right, plan, decisions):
    if not isinstance(plan, dict) or plan.get('schema') != PLAN_SCHEMA:
        raise ValueError('unsupported adjudication plan')
    expected = build_adjudication(left, right, plan.get('reporting_tolerance_px'))
    if expected != plan:
        raise ValueError('reference or plan changed; rebuild and reconfirm')
    if expected['status'] == 'pending_reference_confirmation':
        raise ValueError('two confirmed independent references required')
    if (not isinstance(decisions, dict) or decisions.get('schema') != DECISIONS_SCHEMA
        or decisions.get('plan_sha256') != document_digest(plan)):
        raise ValueError('decision plan hash changed or unsupported decision schema')
    annotator = decisions.get('annotator_id')
    if decisions.get('confirmed') is not True or not isinstance(annotator, str) or not annotator.strip():
        raise ValueError('adjudicator confirmation and ID required')
    resolutions = decisions.get('labels')
    if not isinstance(resolutions, dict): raise ValueError('decision labels must be an object')
    known = {row['key'] for row in expected['rows']}
    if set(resolutions) - known: raise ValueError('decision outside declared label set')
    frames = {f['frame_id']: f for f in expected['frames']}
    labels = {}
    for row in expected['rows']:
        decision = resolutions.get(row['key'])
        if decision is None:
            if row['review_required']: raise ValueError('unresolved disagreement: ' + row['key'])
            point = copy.deepcopy(row['left'])
            origin, choice, reason = 'human_agreement', 'exact_agreement', 'same visibility and coordinates'
        else:
            if not isinstance(decision, dict): raise ValueError('invalid adjudication decision')
            choice, reason = decision.get('choice'), decision.get('reason')
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError('source-review reason required')
            if choice in ('left', 'right'):
                if row[choice] is None: raise ValueError('cannot accept a missing annotation')
                point = copy.deepcopy(row[choice])
            elif choice == 'custom': point = decision.get('point')
            else: raise ValueError('choose left, right or an independently reviewed custom point')
            point = _point(point, frames[row['frame_id']])
            origin = 'human_adjudicated'
        point.update(origin=origin, reviewed=True, adjudication_choice=choice,
                     adjudication_reason=reason, adjudicator_id=annotator.strip())
        labels[row['key']] = point
    return {'schema': 'tennis.independent-joint-labels.v1', 'source_sha256': expected['source_sha256'],
            'coordinate_space': expected['coordinate_space'], 'frame_index_base': 0,
            'requested_joints': expected['requested_joints'], 'frames': expected['frames'],
            'annotator_id': annotator.strip(), 'confirmed': True,
            'annotation_mode': 'independent_adjudicated', 'independent_reference': True,
            'labels': labels, 'accuracy_validated': False,
            'adjudication': {'plan_sha256': document_digest(plan),
                             'decisions_sha256': document_digest(decisions),
                             'references': expected['references'],
                             'method': 'exact agreement plus explicit human decisions; no coordinate averaging',
                             'limitations': expected['limitations']}}
