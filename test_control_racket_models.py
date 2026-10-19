import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import yaml
from local_control_panel import ControlSettings
from racket_model_profiles import EXPERIMENT_640_PATH, apply_model_profile


class ControlRacketModelTests(unittest.TestCase):
    def test_640_requires_explicit_shadow_and_continuous_processing(self):
        base = {'stream_id': 'local_video', 'racket_model_profile': '640_shadow'}
        for mode in ('legacy', 'active', 'exclusive'):
            with self.assertRaises(ValueError):
                ControlSettings.from_payload({**base, 'racket_pose_mode': mode})
        for processing in ('auto', 'realtime'):
            with self.assertRaises(ValueError):
                ControlSettings.from_payload({**base, 'racket_pose_mode': 'shadow', 'processing_mode': processing})
        settings = ControlSettings.from_payload({**base, 'racket_pose_mode': 'shadow', 'processing_mode': 'buffered'})
        self.assertEqual(settings.racket_model_profile, '640_shadow')

    def test_profile_uses_independent_candidate_and_cannot_borrow_quality_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / EXPERIMENT_640_PATH).mkdir(parents=True)
            config = {'racket_pose': {'model_path': 'original', 'acceptance_path': '960-receipt'}, 'roi_settings': {'enabled': True}}
            saved = copy.deepcopy(config)
            apply_model_profile(config, 'configured', 'exclusive', root)
            self.assertEqual(config, saved)
            with self.assertRaises(ValueError):
                apply_model_profile(config, '640_shadow', 'exclusive', root)
            apply_model_profile(config, '640_shadow', 'shadow', root)
            self.assertEqual(config['racket_pose']['imgsz'], 640)
            self.assertEqual(config['racket_pose']['compute_units'], 'ALL')
            self.assertIsNone(config['racket_pose']['acceptance_path'])
            self.assertEqual(config['roi_settings'], saved['roi_settings'])

    def test_actual_start_writes_640_runtime_and_reports_startup_identity(self):
        from test_local_control_panel import LocalControlPanelTests
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            controller = LocalControlPanelTests().make_controller(root)
            (root / EXPERIMENT_640_PATH).mkdir(parents=True)
            original_config = controller.config_path.read_bytes()
            process = Mock(pid=999999)
            process.poll.return_value = None
            with patch('local_control_panel.subprocess.Popen', return_value=process), patch('local_control_panel.threading.Thread'):
                status = controller.start({'stream_id': 'court01-main', 'racket_model_profile': '640_shadow',
                                           'racket_pose_mode': 'shadow', 'processing_mode': 'buffered'})
            try:
                config = yaml.safe_load((controller._runtime_directory / 'runtime_config.yaml').read_text())
                self.assertEqual(config['racket_pose']['imgsz'], 640)
                self.assertEqual(config['racket_pose']['mode'], 'shadow')
                self.assertEqual(status['racket_model']['imgsz'], 640)
                self.assertEqual(status['racket_model']['profile'], '640_shadow')
                self.assertTrue(any('640x640' in line for line in status['logs']))
                self.assertEqual(controller.config_path.read_bytes(), original_config)
            finally:
                shutil.rmtree(controller._runtime_directory)


if __name__ == '__main__':
    unittest.main()
