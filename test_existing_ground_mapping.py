import copy
import unittest
from ground_reference import normalize_calibration
from test_ground_reference import calibration
from scripts.audit_existing_ground_mapping import mapping_reuse_report


class MappingReuseTests(unittest.TestCase):
    def test_reflection_keeps_labels_and_same_metre_coordinates(self):
        cal=normalize_calibration(calibration());before=copy.deepcopy(cal)
        report=mapping_reuse_report(cal)
        front,back=report['groups']
        self.assertEqual([c['label'] for c in back['corners']],["A′","B′","C′","D′"])
        for a,b in zip(front['corners'],back['corners']):
            self.assertEqual(a['world_m'],b['world_m'])
            self.assertLess(a['fit_residual_m'],1e-10)
            self.assertLess(b['fit_residual_m'],1e-10)
        self.assertEqual(cal,before)
        self.assertFalse(report['independent_scale_accuracy_validated'])

    def test_changed_frozen_geometry_rejected(self):
        cal=normalize_calibration(calibration());cal['width_m']=4
        with self.assertRaises(ValueError):mapping_reuse_report(cal)

    def test_machine_epsilon_recompute_keeps_frozen_identity(self):
        import hashlib,json,math
        cal=normalize_calibration(calibration())
        cal['views']['back']['H'][0][0]=math.nextafter(cal['views']['back']['H'][0][0],math.inf)
        frozen={k:v for k,v in cal.items() if k!='calibration_id'}
        cal['calibration_id']=hashlib.sha256(json.dumps(frozen,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        report=mapping_reuse_report(cal)
        self.assertEqual(report['calibration_id'],cal['calibration_id'])

    def test_missing_mirror_rejected(self):
        cal=calibration();del cal['views']['back'];cal=normalize_calibration(cal)
        with self.assertRaises(ValueError):mapping_reuse_report(cal)


if __name__=='__main__':unittest.main()
