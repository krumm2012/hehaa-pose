import unittest
from unittest.mock import patch

from local_control_panel import ControlSettings
from main_pipe import build_argument_parser
from scripts.export_racket_pose_mixed import bounded_pose_decode


class InterfaceTests(unittest.TestCase):
    def test_defaults_preserve_old_workflow(self):
        args=build_argument_parser().parse_args([])
        self.assertIsNone(args.processing_mode);self.assertIsNone(args.racket_pose_mode)
        settings=ControlSettings.from_payload({'stream_id':'local_video'})
        self.assertEqual(settings.racket_pose_mode,'legacy');self.assertEqual(settings.processing_mode,'buffered')
        self.assertFalse(settings.live_mode)

    def test_local_realtime_is_explicit_and_offline_network_rejected(self):
        settings=ControlSettings.from_payload({'stream_id':'local_video','processing_mode':'realtime','racket_pose_mode':'shadow'})
        self.assertEqual(settings.processing_mode,'realtime');self.assertEqual(settings.racket_pose_mode,'shadow')
        with self.assertRaises(ValueError):ControlSettings.from_payload({'stream_id':'court01-main','processing_mode':'offline'})
        with self.assertRaises(ValueError):ControlSettings.from_payload({'racket_pose_mode':'unknown'})

    def test_decoder_bounds_extremes_without_moving_visible_points(self):
        import torch
        from types import SimpleNamespace
        head=SimpleNamespace(kpt_shape=[7,3],nk=21,anchors=torch.tensor([[60.],[60.]]),strides=torch.tensor([[16.]]))
        raw=torch.zeros((1,21,1));raw[:,0]=45607;raw[:,1]=-45607
        result=bounded_pose_decode(head,raw.half())
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertGreater(float(result[0,0]),960);self.assertLess(float(result[0,1]),0)
        normal=torch.zeros_like(raw);normal[:,0]=.1;normal[:,1]=-.2
        expected=(normal.view(1,7,3,1)[:,:,:2]*2+(head.anchors-.5))*head.strides
        actual=bounded_pose_decode(head,normal).view(1,7,3,1)[:,:,:2]
        torch.testing.assert_close(actual,expected)

if __name__=='__main__':unittest.main()
