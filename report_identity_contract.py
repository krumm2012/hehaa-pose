"""Reports retain declared identity and missing anchors instead of borrowing them."""
from evaluation_identity_contract import normalize_model_event_document

POLICY_VERSION = 'report_identity_v1_strict_event_and_coach_ids'


def normalize_report_document(document, label='报告事件'):
    try:
        result = normalize_model_event_document(document)
    except ValueError as error:
        raise ValueError(f'{label}: {error}') from error
    for event in result['events']:
        # Model aliases use root anchors first; an explicit null stays missing.
        if 'peak_frame' not in event:
            frames = event['frames']
            event['peak_frame'] = (frames['peak_frame'] if 'peak_frame' in frames
                                   else frames.get('peak'))
    return result


def report_identity_info():
    return {'policy_version': POLICY_VERSION,
            'event_identity': 'nonnegative_safe_integer_no_coercion_unique',
            'coach_join': 'declared_event_id_only',
            'source_binding_verified': False,
            'missing_anchor_policy': 'preserve_missing_no_coach_fallback',
            'accuracy_validated': False}
