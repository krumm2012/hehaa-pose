"""Check a reused ABCD / A'B'C'D' floor map; no independent accuracy claim."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ground_reference import normalize_calibration, map_point


def mapping_reuse_report(document):
    cal = normalize_calibration(document)
    # Verify the frozen bytes' semantic fingerprint, not a newly solved matrix's
    # fingerprint: LAPACK environments can differ at machine epsilon.
    frozen = {k:v for k,v in document.items() if k != 'calibration_id'}
    fingerprint = hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if document.get('calibration_id') != fingerprint:
        raise ValueError('Frozen calibration fingerprint mismatch')
    for view, data in cal['views'].items():
        if not np.allclose(document['views'][view]['H'], data['H'], rtol=1e-12, atol=1e-12):
            raise ValueError('Frozen matrix differs from declared geometry')
        data['H'] = document['views'][view]['H']
    cal['calibration_id'] = document['calibration_id']
    if set(cal['views']) != {'front', 'back'}:
        raise ValueError('Both ABCD and reflected correspondences required')
    world = [[0,0],[cal['width_m'],0],[cal['width_m'],cal['length_m']],[0,cal['length_m']]]
    groups = []
    for view, data in cal['views'].items():
        inverse = np.linalg.inv(data['H'])
        corners = [{'label': name+("′" if view=='back' else ''), 'image': pixel,
                    'world_m': target, 'mapped_world_m': map_point(data['H'],pixel),
                    'fit_residual_m': math.dist(map_point(data['H'],pixel),target)}
                   for name,pixel,target in zip(data['corner_ids'],data['points'],world)]
        errors = []
        for fx in (.1,.5,.9):
            for fy in (.1,.5,.9):
                target = [cal['width_m']*fx,cal['length_m']*fy]
                pixel = map_point(inverse,target)
                errors.append(math.dist(map_point(data['H'],pixel),target))
        groups.append({'view':view,'H':data['H'],'corners':corners,
                       'max_grid_roundtrip_residual_m':max(errors)})
    return {'schema':'tennis.existing-ground-mapping-audit.v1',
            'status':'existing_corner_mapping_verified', 'calibration_id':cal['calibration_id'],
            'binding':cal['binding'],'width_m':cal['width_m'],'length_m':cal['length_m'],
            'axis_convention':'A / A′ origin; AB / A′B′ positive X; AD / A′D′ positive Y',
            'correspondence_confirmed':cal['correspondence_confirmed'],
            'dimensions_measured':cal['dimensions_measured'],'groups':groups,
            'independent_scale_accuracy_validated':False,'coaching_eligible':False,
            'residual_semantics':'Four-point fit and round-trip numerics, not held-out physical error',
            'scope':'Floor-plane mapping only; body height/3D rotation require other evidence'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();raw=Path(a.calibration).read_bytes()
    result=mapping_reuse_report(json.loads(raw));result['calibration_sha256']=hashlib.sha256(raw).hexdigest()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as file: file.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(result['status'])


if __name__=='__main__':main()
