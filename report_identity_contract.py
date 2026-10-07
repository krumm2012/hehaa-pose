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


def verify_source_session_binding(event_document: dict, coach_document: dict) -> bool:
    """确保同一份报告内的事件与教练数据来自相同录像会话"""
    if not isinstance(event_document, dict) or not isinstance(coach_document, dict):
        return False
    event_source = event_document.get('source') or {}
    coach_source = coach_document.get('source') or {}

    # 1. 校验会话特征 session_id
    event_session = event_source.get('session_id') or event_document.get('session_id')
    coach_session = coach_source.get('session_id') or coach_document.get('session_id')
    if event_session and coach_session and event_session != coach_session:
        raise ValueError(
            f"跨会话冲突拒绝：模型事件会话为 '{event_session}'，但 Coach 输入会话为 '{coach_session}'"
        )

    # 2. 校验源视频内容特征 video_sha256
    event_sha = event_source.get('video_sha256') or event_document.get('video_sha256')
    coach_sha = coach_source.get('video_sha256') or coach_document.get('video_sha256')
    if event_sha and coach_sha and event_sha != coach_sha:
        raise ValueError(
            f"源视频 SHA 不符：模型事件哈希 '{str(event_sha)[:8]}...' 与 Coach 哈希 '{str(coach_sha)[:8]}...' 不一致"
        )

    # 3. 校验源视频路径 video_path（若均显式存在）
    event_vpath = event_source.get('video_path') or event_document.get('video_path')
    coach_vpath = coach_source.get('video_path') or coach_document.get('video_path')
    if event_vpath and coach_vpath and str(event_vpath).strip() != str(coach_vpath).strip():
        raise ValueError(
            f"源视频路径不符：模型事件视频为 '{event_vpath}'，但 Coach 视频为 '{coach_vpath}'"
        )

    return bool(
        (event_session and coach_session and event_session == coach_session) or
        (event_sha and coach_sha and event_sha == coach_sha) or
        (event_vpath and coach_vpath and str(event_vpath).strip() == str(coach_vpath).strip())
    )


def report_identity_info(source_binding_verified: bool = False):
    return {'policy_version': POLICY_VERSION,
            'event_identity': 'nonnegative_safe_integer_no_coercion_unique',
            'coach_join': 'declared_event_id_with_session_binding' if source_binding_verified else 'declared_event_id_only',
            'source_binding_verified': bool(source_binding_verified),
            'missing_anchor_policy': 'preserve_missing_no_coach_fallback',
            'accuracy_validated': False}
