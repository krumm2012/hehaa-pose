"""Re-estimate corrupt BN buffers on TRAIN ONLY; preserve all learned parameters.

This creates a NEW candidate, never edits the supplied checkpoint or package.
Validation uses the existing correlated split, not independent accuracy proof.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--train-images', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--repair-only-invalid', action='store_true',
                        help='Keep every finite original buffer; replace only lost BN statistics')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(6)
    yolo = YOLO(args.source)
    model = yolo.model.float().cpu()
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    original_buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    bad = {k: int((~torch.isfinite(v)).sum()) for k, v in model.state_dict().items() if not torch.isfinite(v).all()}
    if any(not torch.isfinite(v).all() for v in before.values()):
        raise ValueError('Non-finite learned parameters cannot be repaired by BN calibration')
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            module.momentum = None
    files = sorted(Path(args.train_images).glob('*.jpg'))
    if not files:
        raise ValueError('No calibration training images')
    letterbox = LetterBox(new_shape=(960, 960), auto=False)
    with torch.no_grad():
        for start in range(0, len(files), 4):
            batch = [letterbox(image=cv2.imread(str(p))) for p in files[start:start+4]]
            tensor = torch.from_numpy(np.stack(batch)[:, :, :, ::-1].transpose(0, 3, 1, 2).copy()).float()/255
            model(tensor)
            if start % 40 == 0:
                print('BN calibration',start,'/',len(files),flush=True)
    model.eval()
    if args.repair_only_invalid:
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                original = original_buffers[name]
                finite = torch.isfinite(original)
                buffer[finite] = original[finite]
    assert all(torch.equal(v, before[k]) for k,v in model.named_parameters())
    nonfinite = {k: int((~torch.isfinite(v)).sum()) for k,v in model.state_dict().items() if not torch.isfinite(v).all()}
    if nonfinite:
        raise ValueError('Recalibration still nonfinite: '+str(nonfinite))
    ckpt = copy.copy(yolo.ckpt)
    ckpt['model'] = copy.deepcopy(model).float()
    ckpt['ema'] = None
    ckpt['optimizer'] = None
    ckpt['train_metrics'] = {}  # Historical fitness does not apply to the new candidate.
    checkpoint = out/('bn_repaired_fp32.pt' if args.repair_only_invalid else 'bn_recalibrated_fp32.pt')
    torch.save(ckpt, checkpoint)
    report = {'source':str(Path(args.source).resolve()),'original_nonfinite_buffers':bad,
              'calibration_split':'train','calibration_images':len(files),
              'learned_parameters_unchanged':True,'nonfinite_after':nonfinite,
              'finite_original_buffers_preserved':args.repair_only_invalid,
              'accuracy_validated':False,'checkpoint':str(checkpoint),
              'reason':'Re-estimate lost BN inference statistics; not a numerically equivalent export.'}
    (out/'calibration.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    if args.export:
        import subprocess
        subprocess.run([sys.executable, str(Path(__file__).with_name('export_racket_pose_mixed.py')),
                        '--checkpoint',str(checkpoint),'--output',str(out/'racket_pose_bounded_fp16.mlpackage'),
                        '--bounded'],check=True)
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
