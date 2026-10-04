"""Identify newly generated analysis without rewriting historical records."""
import hashlib
from pathlib import Path

from kinematic_sequence import POLICY_VERSION as KINEMATIC_POLICY
from metric_contracts import VERSION as METRIC_CONTRACT
from osd_evidence import POLICY as OBSERVATION_POLICY
from swing_coach_calibration import CALIBRATION_POLICY_VERSION
from event_source_timing import POLICY_VERSION as PHASE_TIME_POLICY
from coach_rule_contract import POLICY_VERSION as AUTOMATIC_COACH_POLICY
from motion_time_contract import POLICY_VERSION as MOTION_TIME_POLICY
from metric_source_windows import POLICY_VERSION as BODY_WINDOW_POLICY
from swing_event_classifier import POLICY_VERSION as CLASSIFICATION_POLICY
from image_motion_measurements import POLICY_VERSION as IMAGE_MOTION_POLICY

_FILES = ('swing_event_analyzer.py','swing_biomechanics.py','kinematic_sequence.py',
          'osd_evidence.py','realtime_swing_pipeline.py','swing_motion_features.py',
          'analysis_metric_delivery.py','metric_contracts.py')
_FILES += ('ball_observation_contract.py','frame_processor.py','main_pipe.py','yolo26n_unified_detector.py')
_FILES += ('event_source_timing.py','local_realtime_coach.py',
           'swing_coach_data_collector.py','coach_evidence_policy.py','deepseek_evidence_view.py')
_FILES += ('coach_rule_contract.py', 'swing_report_builder.py', 'analysis_provenance.py',
           'deepseek_realtime_coach.py')
_FILES += ('manual_review_workflow.py',)
_FILES += ('swing_event_segmenter.py', 'swing_event_classifier.py',
           'motion_time_contract.py', 'image_motion_measurements.py')
_FILES += ('session_evidence_bundle.py', 'swing_quality_policy.py')
_FILES += ('metric_source_windows.py',)
_HASHES = {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in _FILES}


def analysis_build_info():
    return {'schema':'tennis.analysis-build.v1', 'code_sha256':dict(_HASHES),
            'hash_semantics':'source files at provenance module import; restart required after source edits',
            'metric_contract':METRIC_CONTRACT,'kinematic_policy':KINEMATIC_POLICY,
            'observation_policy':OBSERVATION_POLICY,'scoring_policy':CALIBRATION_POLICY_VERSION,
            'window_policy':'realtime_source_time_and_capacity_separated_v3',
            'ball_observation_contract':'tennis.ball-observation.v1',
            'phase_time_policy':PHASE_TIME_POLICY,
            'automatic_coach_policy':AUTOMATIC_COACH_POLICY,
            'motion_time_policy':MOTION_TIME_POLICY,
            'body_window_policy':BODY_WINDOW_POLICY,
            'classification_policy':CLASSIFICATION_POLICY,
            'image_motion_policy':IMAGE_MOTION_POLICY,
            'historical_outputs_rewritten':False, 'accuracy_validated':False}
