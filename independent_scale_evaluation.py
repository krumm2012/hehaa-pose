"""Evaluate held-out measured floor points against a frozen calibration."""
import math
from collections import defaultdict
from statistics import mean

import numpy as np
from ground_reference import normalize_calibration, map_point
from observation_policy import finite_number


def evaluate_scale_checks(review, calibration, calibration_sha256, tolerance_m):
    tolerance = finite_number(tolerance_m)
    if tolerance is None or tolerance <= 0:
        raise ValueError('Explicit positive tolerance in metres required')
    if review.get('schema') != 'tennis.independent-scale-check-draft.v1':
        raise ValueError('Unsupported scale review')
    if (review.get('calibration_sha256') != calibration_sha256
            or review.get('calibration_id') != calibration.get('calibration_id')):
        raise ValueError('Frozen calibration identity mismatch')
    if (review.get('binding') != calibration.get('binding')
            or review.get('source_sha256') != calibration.get('binding', {}).get('source_id')
            or calibration.get('binding', {}).get('kind') != 'video_sha256'):
        raise ValueError('Source identity mismatch')
    if (review.get('coordinate_space') != 'original_source_pixels'
            or review.get('frame_index_base') != 0
            or type(review.get('frame_id')) is not int or review['frame_id'] < 0
            or review.get('image_size') != calibration.get('image_size')):
        raise ValueError('Source coordinate contract mismatch')
    base = {'schema': 'tennis.independent-scale-evaluation.v1',
            'source_sha256': review['source_sha256'], 'calibration_id': review['calibration_id'],
            'calibration_sha256': calibration_sha256, 'tolerance_m': tolerance,
            'tolerance_semantics': 'explicit_evaluation_parameter_not_coaching_standard',
            'accuracy_validated': False, 'coaching_eligible': False,
            'historical_calibration_modified': False}
    if review.get('confirmed') is not True:
        return {**base, 'status': 'awaiting_physical_measurements', 'groups': []}
    for name in ('annotator_id', 'instrument', 'measurement_evidence'):
        if not isinstance(review.get(name), str) or not review[name].strip():
            raise ValueError('Missing measurement provenance: '+name)
    uncertainty = finite_number(review.get('measurement_uncertainty_m'))
    if uncertainty is None or uncertainty < 0:
        raise ValueError('Measurement uncertainty required')
    checks = review.get('check_points')
    if not isinstance(checks, list) or not checks:
        raise ValueError('Confirmed review requires measured points')
    frozen = normalize_calibration(calibration)
    groups = defaultdict(list)
    seen, ids = set(), set()
    for point in checks:
        view = point.get('view')
        if view not in frozen['views']:
            raise ValueError('Unknown view')
        if not isinstance(point.get('id'), str) or not point['id'] or point['id'] in ids:
            raise ValueError('Missing or duplicate check identity')
        ids.add(point['id'])
        values = []
        for name in ('image', 'world_m'):
            xy = point.get(name)
            if not isinstance(xy, list) or len(xy) != 2:
                raise ValueError('Invalid '+name)
            xy = [finite_number(v) for v in xy]
            if any(v is None for v in xy): raise ValueError('Nonfinite '+name)
            values.append(xy)
        image, actual = values
        w, h = frozen['image_size']
        if not 0 <= image[0] < w or not 0 <= image[1] < h:
            raise ValueError('Image point outside source')
        identity = (view, *image)
        if identity in seen: raise ValueError('Duplicate source point')
        seen.add(identity)
        if any(math.dist(image, corner) < 1 for corner in frozen['views'][view]['points']):
            raise ValueError('Fit corners are not held-out checks')
        projected = map_point(frozen['views'][view]['H'], image)
        reprojection = map_point(np.linalg.inv(frozen['views'][view]['H']), actual)
        error = math.dist(projected, actual)
        groups[view].append({'id': point['id'], 'image': image, 'world_m': actual,
            'predicted_world_m': projected, 'error_m': error,
            'reprojection_error_px': math.dist(reprojection, image),
            'conservative_error_m': error+uncertainty})
    summaries = []
    for view, rows in groups.items():
        # Need distributed checks; three collinear checks cannot test both axes.
        xyz = np.asarray([r['world_m'] for r in rows])
        coverage = len(rows) >= 3 and np.linalg.matrix_rank(xyz-xyz.mean(axis=0)) == 2
        summaries.append({'view': view, 'point_count': len(rows),
            'distributed_check_coverage': bool(coverage),
            'mean_error_m': mean(r['error_m'] for r in rows),
            'max_error_m': max(r['error_m'] for r in rows),
            'evaluation_passed': bool(coverage and all(r['conservative_error_m'] <= tolerance for r in rows)),
            'samples': rows})
    return {**base, 'status': 'evaluated_measured_checks',
            'measurement_uncertainty_m': uncertainty, 'groups': summaries,
            'scope': 'measured_floor_points_only_no_body_height_or_3d_accuracy',
            'untested_views': sorted(set(frozen['views'])-set(groups))}
