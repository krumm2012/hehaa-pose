"""Same-input PyTorch/Core ML checks, without granting a quality acceptance."""
import argparse
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect
    from racket_pose_detector import RacketPoseDetector, letterbox, model_digest, decode_candidates
    from scripts.export_racket_pose_mixed import bounded_pose_decode
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    reference = YOLO(args.checkpoint).model.float().cpu().eval().fuse()
    for module in reference.modules():
        if isinstance(module, Detect):
            module.export = True
            module.format = 'coreml'
            module.dynamic = False
            module.shape = None
    reference.model[-1].kpts_decode = types.MethodType(bounded_pose_decode, reference.model[-1])
    with torch.no_grad():
        reference(torch.zeros(1, 3, 640, 640))
    models = {u: RacketPoseDetector({'mode': 'shadow', 'model_path': args.candidate,
                                    'imgsz': 640, 'compute_units': u}) for u in ('ALL', 'ANE')}
    rows = []
    for video in sorted(Path('/Users/krum5539/Desktop/Camera/2026-09-30').glob('*.mp4')):
        cap = cv2.VideoCapture(str(video))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError('Source decode failed')
        rgb, transform = letterbox(frame, 640)
        with torch.no_grad():
            output = reference(torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None] / 255)
        pt = (output[0] if isinstance(output, tuple) else output).numpy()
        for units, detector in models.items():
            ml = detector.model.predict({'image': Image.fromarray(rgb)})[detector.output_name]
            finite = bool(np.isfinite(pt).all() and np.isfinite(ml).all())
            selected = pt[0, 4] >= .25
            pt_points = pt[0, 5:].T[selected].reshape(-1, 7, 3)
            ml_points = ml[0, 5:].T[selected].reshape(-1, 7, 3)
            mask = pt_points[:, :, 2] >= .5
            errors = np.linalg.norm(pt_points[:, :, :2] - ml_points[:, :, :2], axis=-1)[mask]
            a, _ = decode_candidates(pt, transform, detector.config)
            b, _ = decode_candidates(ml, transform, detector.config)
            rows.append({'video': video.name, 'frame_id': 0, 'compute_units': units, 'raw_finite': finite,
                         'point_error_p95_input_pixels': float(np.percentile(errors, 95)) if len(errors) else None,
                         'pt_candidates': len(a), 'ml_candidates': len(b),
                         'confidence_threshold_crossings': int(np.count_nonzero((pt[0, 4] >= .25) != (ml[0, 4] >= .25)))})
        print('PARITY', video.name, flush=True)
    result = {'model_sha256': model_digest(args.candidate), 'reference_sha256': model_digest(args.checkpoint),
              'imgsz': 640, 'rows': rows, 'scope': '9 same-pixel first frames per backend; numerical/export comparison only, not a dataset accuracy gate.',
              'quality_gate_passed': False, 'accuracy_validated': False}
    Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
