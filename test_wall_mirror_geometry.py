import unittest
import numpy as np
from ground_reference import map_point
from scripts.audit_wall_mirror_geometry import mirror_rectangle_diagnostic

class WallMirrorTests(unittest.TestCase):
    def test_reflected_floor_preserves_rectangle_under_perspective(self):
        g=np.array([[130,20,900],[5,80,650],[.01,.02,1.]])
        mirror=[map_point(g,p) for p in [[0,-2],[3.3,-2],[3.3,-6.8],[0,-6.8]]]
        r=mirror_rectangle_diagnostic(np.linalg.inv(g),mirror,3.3,4.8)
        for s in r['side_checks']:self.assertAlmostEqual(s['difference_m'],0,places=9)
        self.assertFalse(r['independent_accuracy_verified'])

    def test_distorted_far_edge_detected_without_refitting(self):
        g=np.array([[100,0,900],[0,60,700],[0,0,1.]])
        mirror=[map_point(g,p) for p in [[0,-2],[3.3,-2],[2.7,-6.8],[0,-6.8]]]
        r=mirror_rectangle_diagnostic(np.linalg.inv(g),mirror,3.3,4.8)
        self.assertAlmostEqual(r['side_checks'][2]['difference_m'],-.6)
        self.assertFalse(r['calibration_modified'])

if __name__=='__main__':unittest.main()
