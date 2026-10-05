"""Conditional vertical-plane-mirror consistency diagnostic; never changes fitting."""
import json
import math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ground_reference import map_point


def mirror_rectangle_diagnostic(front_h, mirror_pixels, width_m, length_m):
    """Extrapolate the direct ground homography into virtual floor, for audit only.

    A vertical ideal mirror reflects the floor into the same extended plane, so
    its virtual rectangle must preserve side lengths. This check cannot isolate
    corner error from lens distortion, mirror tilt/warping or direct-H error.
    """
    virtual=[map_point(front_h,p) for p in mirror_pixels]
    expected=[width_m,length_m,width_m,length_m]
    measured=[math.dist(virtual[i],virtual[(i+1)%4]) for i in range(4)]
    return {'virtual_corners_in_direct_ground_basis_m':virtual,
            'side_checks':[{'side':name,'expected_m':want,'extrapolated_m':got,'difference_m':got-want,
                            'relative_difference_percent':100*(got/want-1)}
                           for name,want,got in zip(('AB','BC','CD','DA'),expected,measured)],
            'conditional_assumptions':['vertical planar mirror','same physical rectangle and correct corner identities',
                                       'shared undistorted pinhole image geometry','valid direct-ground homography outside its fitted patch'],
            'independent_accuracy_verified':False,'calibration_modified':False}
