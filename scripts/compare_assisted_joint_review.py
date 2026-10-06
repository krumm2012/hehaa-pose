"""Compare inference scales with reviewed model suggestions; never report accuracy."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


def compare(review, predictions):
    if (review.get('schema') != 'tennis.assisted-joint-review.v1'
            or review.get('independent_reference') is not False
            or review.get('confirmed') is not True
            or not review.get('annotator_id')
            or review.get('coordinate_space') != 'original_source_pixels'
            or review.get('frame_index_base') != 0):
        raise ValueError('Confirmed assisted review in source pixels required')
    if review['source_sha256'] != predictions.get('source_sha256'):
        raise ValueError('Source hash mismatch')
    frames = {f['frame_id']: f for f in review['frames']}
    if review.get('require_complete_review'):
        joints = review.get('requested_joints')
        if not isinstance(joints, list) or not joints or len(set(joints)) != len(joints):
            raise ValueError('Invalid full-review joint set')
        expected = {f'{fid}:{view}:{joint}' for fid in frames for view in ('front','back') for joint in joints}
        if set(review['labels']) != expected:
            raise ValueError('Full review requires every declared frame/view/joint')

    groups = []
    for scale in predictions['scales']:
        for view in ('front', 'back'):
            rows = []
            for key, target in review['labels'].items():
                fid, target_view, joint = key.split(':')
                fid = int(fid)
                if target_view != view:
                    continue
                if target.get('reviewed') is not True:
                    raise ValueError('Unreviewed label present')
                if not target['visible']:
                    continue
                frame = frames[fid]
                def valid_xy(point):
                    return all(type(point.get(k)) in (int, float)
                               and math.isfinite(point[k]) and 0 <= point[k] < frame[dim]
                               for k, dim in (('x', 'width'), ('y', 'height')))
                if not valid_xy(target):
                    raise ValueError('Invalid accepted coordinates')
                point = predictions['samples'].get(str(fid), {}).get(str(scale), {}).get(view, {}).get(joint, {})
                qualified = (point.get('observed') is True
                             and type(point.get('source_frame_id')) is int
                             and point['source_frame_id'] == fid
                             and type(point.get('confidence')) in (int, float)
                             and math.isfinite(point['confidence']) and point['confidence'] >= .5
                             and valid_xy(point))
                distance = math.hypot(point['x']-target['x'], point['y']-target['y']) if qualified else None
                rows.append({'key': key, 'qualified': qualified, 'displacement_px': distance})
            distances = [r['displacement_px'] for r in rows if r['qualified']]
            groups.append({'scale': scale, 'view': view, 'accepted_visible_count': len(rows),
                           'qualified_count': len(distances), 'missing_count': len(rows)-len(distances),
                           'mean_displacement_px': mean(distances) if distances else None,
                           'median_displacement_px': median(distances) if distances else None,
                           'max_displacement_px': max(distances) if distances else None,
                           'samples': sorted(rows, key=lambda r: r['displacement_px'] if r['qualified'] else float('inf'), reverse=True)})
    return {'schema': 'tennis.assisted-review-consistency.v1', 'independent_accuracy': None,
            'kinematic_accuracy': None, 'groups': groups,
            'limitations': ['Model-assisted acceptance is not independent ground truth.',
                           'Native-scale zero distance is circular when unchanged suggestions were accepted.',
                           'Six selected frames do not measure temporal peaks or whole-video accuracy.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('review', 'predictions', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    review_bytes = Path(args.review).read_bytes()
    prediction_bytes = Path(args.predictions).read_bytes()
    review = json.loads(review_bytes)
    if review['prediction_sha256'] != hashlib.sha256(prediction_bytes).hexdigest():
        raise ValueError('Prediction hash mismatch')
    result = compare(review, json.loads(prediction_bytes))
    result['review_sha256'] = hashlib.sha256(review_bytes).hexdigest()
    result['prediction_sha256'] = hashlib.sha256(prediction_bytes).hexdigest()
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps([{k: v for k, v in g.items() if k != 'samples'} for g in result['groups']], indent=2))


if __name__ == '__main__':
    main()
