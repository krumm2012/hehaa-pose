"""Release identity must reject missing, changed or promoted model contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from racket_model_profiles import EXPERIMENT_640_PATH
from racket_pose_detector import model_digest
from scripts.verify_shadow_release import DEFAULT_CONTRACT, verify


class ShadowReleasePackageTests(unittest.TestCase):
    def test_manifest_matches_installed_candidate(self):
        root = Path(__file__).resolve().parent
        if not (root / EXPERIMENT_640_PATH).is_dir():
            self.skipTest('Release model not installed')
        self.assertTrue(verify()['model_verified'])

    def test_missing_and_modified_package_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = json.loads(DEFAULT_CONTRACT.read_text())
            path = root / 'contract.json'
            path.write_text(json.dumps(contract))
            with self.assertRaises(FileNotFoundError):
                verify(root, path)
            model = root / EXPERIMENT_640_PATH
            model.mkdir(parents=True)
            (model / 'weights.bin').write_bytes(b'original')
            contract['model']['sha256'] = model_digest(model)
            path.write_text(json.dumps(contract))
            self.assertTrue(verify(root, path)['model_verified'])
            (model / 'weights.bin').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                verify(root, path)

    def test_quality_promotion_or_wrong_interface_is_rejected(self):
        base = json.loads(DEFAULT_CONTRACT.read_text())
        changes = [('active_or_exclusive_allowed', True), ('quality_gate_passed', True),
                   ('not_an_active_acceptance_receipt', False)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'contract.json'
            for key, value in changes:
                with self.subTest(key=key):
                    contract = copy.deepcopy(base)
                    contract[key] = value
                    path.write_text(json.dumps(contract))
                    with self.assertRaisesRegex(ValueError, 'promoted'):
                        verify(directory, path)
            base['model']['imgsz'] = 960
            path.write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError, 'interface'):
                verify(directory, path)
