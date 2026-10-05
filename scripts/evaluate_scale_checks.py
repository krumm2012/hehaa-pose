"""Validate physical check provenance and evaluate without updating calibration."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_scale_evaluation import evaluate_scale_checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('review', 'calibration', 'source', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--tolerance-m', type=float, required=True)
    a = p.parse_args()
    raw = Path(a.calibration).read_bytes()
    review_bytes = Path(a.review).read_bytes()
    review = json.loads(review_bytes)
    source_digest = hashlib.sha256()
    with Path(a.source).open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''): source_digest.update(block)
    if source_digest.hexdigest() != review.get('source_sha256'):
        raise ValueError('Actual source video hash mismatch')
    result = evaluate_scale_checks(review, json.loads(raw), hashlib.sha256(raw).hexdigest(), a.tolerance_m)
    # Confirm the clicked image is exactly the declared source frame.
    import cv2
    cap = cv2.VideoCapture(a.source)
    try:
        for _ in range(review['frame_id']+1):
            ok, image = cap.read()
            if not ok: raise ValueError('Missing source frame')
    finally: cap.release()
    if list(image.shape[1::-1]) != review['image_size']: raise ValueError('Source dimensions mismatch')
    ok, png = cv2.imencode('.png', image)
    if not ok or hashlib.sha256(png.tobytes()).hexdigest() != review.get('source_frame_png_sha256'):
        raise ValueError('Source frame pixels mismatch')
    result['review_sha256'] = hashlib.sha256(review_bytes).hexdigest()
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as file: file.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(result['status'])


if __name__ == '__main__': main()
