"""Explicit control-panel model choices; experimental quality gates stay intact."""
from pathlib import Path

EXPERIMENT_640_PATH = 'models/racket_pose/candidates/bn-repaired-640-v1/racket_pose_bounded_fp16_640.mlpackage'


def apply_model_profile(config, profile, mode, workspace):
    if profile == 'configured':
        return
    if profile != '640_shadow':
        raise ValueError('未知的球拍模型配置')
    if mode != 'shadow':
        raise ValueError('640候选尚未通过检测质量验收，仅支持影子性能预览')
    path = Path(workspace) / EXPERIMENT_640_PATH
    if not path.is_dir():
        raise ValueError('640实验候选不存在，请先完成独立候选导出')
    config.setdefault('racket_pose', {}).update(
        model_path=str(path.resolve()), imgsz=640, backend='coreml', compute_units='ALL',
        mode='shadow', selection_policy='mixed', acceptance_run=False,
        use_accepted_schedule=False, inference_stride=1,
        acceptance_path=None, runtime_acceptance_path=None,
    )
