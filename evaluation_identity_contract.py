"""Validate evaluation identities before matching or writing any output.

Canonical copies reconcile documented aliases without changing the source files.
An explicit null is missing evidence, never permission to borrow another alias.
"""
import math

from manual_annotation_contract import require_manual_frame_id

POLICY_VERSION = 'evaluation_identity_v1_strict_source_frames'
_FRAME_NAMES = ('start', 'contact', 'end')


def _identity(value, label):
    try:
        return require_manual_frame_id(value)
    except ValueError as error:
        raise ValueError(f'{label}: {error}') from error


def _first_declared(fields):
    for mapping, key in fields:
        if key in mapping:
            return mapping[key]
    return None


def _normalize_row(row, label, model):
    if not isinstance(row, dict):
        raise ValueError(f'{label}必须是对象')
    frames = row.get('frames')
    if frames is None:
        frames = {}
    if not isinstance(frames, dict):
        raise ValueError(f'{label}.frames必须是对象')
    # Validate even shadowed aliases: malformed declared identities cannot disappear.
    for name in _FRAME_NAMES + ('peak',):
        for mapping, key in ((row, f'{name}_frame'), (frames, name),
                             (frames, f'{name}_frame')):
            if key in mapping and mapping[key] is not None:
                _identity(mapping[key], f'{label}.{key}')
    canonical = {}
    for name in _FRAME_NAMES:
        primary = ((row, f'{name}_frame'), (frames, f'{name}_frame'), (frames, name))
        if not model:
            primary = ((frames, name), (row, f'{name}_frame'), (frames, f'{name}_frame'))
        canonical[name] = _first_declared(primary)
    start, contact, end = (canonical[name] for name in _FRAME_NAMES)
    if ((start is not None and contact is not None and start > contact)
            or (contact is not None and end is not None and contact > end)
            or (start is not None and end is not None and start > end)):
        raise ValueError(f'{label}帧号顺序必须满足开始帧 ≤ 触球帧 ≤ 结束帧')
    result = dict(row)
    result['frames'] = {**frames, **canonical}
    result.update({f'{name}_frame': value for name, value in canonical.items()})
    for key in ('event_id', 'source_event_id'):
        if key in row and row[key] is not None:
            _identity(row[key], f'{label}.{key}')
    for key in ('valid_hit', 'needs_review', 'count_correct'):
        if key in row and type(row[key]) is not bool:
            raise ValueError(f'{label}.{key}必须是布尔值')
    quality = row.get('quality_flags')
    if quality is not None and not isinstance(quality, dict):
        raise ValueError(f'{label}.quality_flags必须是对象')
    return result


def _normalize_document(document, model, v2):
    label = '模型事件' if model else '参考标注'
    if not isinstance(document, dict):
        raise ValueError(f'{label}文档必须是对象')
    rows = document.get('events', [])
    if not isinstance(rows, list):
        raise ValueError(f'{label}.events必须是数组')
    source = document.get('source')
    if source is not None and not isinstance(source, dict):
        raise ValueError(f'{label}.source必须是对象')
    identities = set()
    normalized = []
    for index, row in enumerate(rows):
        row_label = f'{label}[{index + 1}]'
        result = _normalize_row(row, row_label, model)
        if model or not v2:
            identity = _identity(result.get('event_id'), f'{row_label}.event_id')
        else:
            identity = result.get('annotation_id')
            if identity is None:
                event_id = result.get('event_id')
                identity = str(event_id) if event_id is not None else f'annotation-{index + 1}'
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError(f'{row_label}.annotation_id必须是非空字符串')
            # Resolve legacy row identity before valid_hit filtering can shift indices.
            result['annotation_id'] = identity
        if identity in identities:
            raise ValueError(f'{label}身份重复: {identity}')
        identities.add(identity)
        normalized.append(result)
    return {**document, 'events': normalized, 'source': source or {}}


def normalize_evaluation_inputs(event_data, annotation_data):
    if not isinstance(annotation_data, dict):
        raise ValueError('参考标注文档必须是对象')
    schema = annotation_data.get('schema_version')
    if schema not in (None, 'swing_manual_annotations_v1', 'swing_manual_annotations_v2'):
        raise ValueError('不支持的参考标注schema_version')
    if ('timeline_review_complete' in annotation_data
            and type(annotation_data['timeline_review_complete']) is not bool):
        raise ValueError('timeline_review_complete必须是布尔值')
    v2 = schema == 'swing_manual_annotations_v2'
    return (_normalize_document(event_data, True, v2),
            _normalize_document(annotation_data, False, v2))


def validate_evaluation_settings(contact_tolerance, match_tolerance, minimum_iou):
    _identity(contact_tolerance, 'contact_tolerance_frames')
    _identity(match_tolerance, 'match_contact_tolerance_frames')
    if (type(minimum_iou) not in (int, float) or not 0 <= minimum_iou <= 1
            or not math.isfinite(minimum_iou)):
        raise ValueError('min_event_iou必须是[0,1]范围内的有限数值')
