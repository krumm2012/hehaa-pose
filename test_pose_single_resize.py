"""Source ROI -> one model resize, with unchanged canonical view coordinates."""
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from dual_view_manager import DualViewManager
from pose_estimator_yolo26 import PoseEstimatorYOLO26
from dual_pose_estimator import DualPoseEstimator
from racket_pose_detector import letterbox
from yolo26n_unified_detector import YOLO26nUnifiedDetector


class SingleResizeTests(unittest.TestCase):
    def estimator(self):
        obj=PoseEstimatorYOLO26.__new__(PoseEstimatorYOLO26)
        obj.input_size=(640,640)
        obj.keypoint_names=['joint'+str(i) for i in range(17)]
        obj.confidence_threshold=.25;obj.keypoint_confidence=.3
        obj.roi_manager=None;obj.pose_smoothing_enabled=False
        output=np.zeros((1,1,57),np.float32);output[0,0,4]=.9
        output[0,0,6:]=np.array([320,320,.9]*17)
        class Model:
            def predict(self,inputs):return {'output':output}
        obj.model=Model()
        return obj

    def test_source_rois_are_not_resized_to_display_sizes(self):
        manager=DualViewManager()
        frame=np.random.default_rng(2).integers(0,256,(1440,2560,3),dtype=np.uint8)
        with patch('dual_view_manager.cv2.resize',wraps=cv2.resize) as resize:
            dual=manager.split_frame(frame,for_inference=True)
        self.assertEqual(resize.call_count,0)
        for image,info,flag in [(dual.front_frame,dual.front_info,dual.front_source_pixels),
                               (dual.back_frame,dual.back_info,dual.back_source_pixels)]:
            self.assertTrue(flag)
            self.assertEqual(image.shape[1::-1],info.crop_size)
            self.assertEqual(info.view_size,(540,720))
        est=self.estimator()
        with patch('pose_estimator_yolo26.cv2.resize',wraps=cv2.resize) as resize:
            for image in (dual.front_frame,dual.back_frame):
                result=est.get_keypoints_from_source(image,coordinate_size=(540,720))
                self.assertEqual(result[0]['joint0'][:2],(270,360))
        self.assertEqual(resize.call_count,2)

    def test_already_model_size_has_no_resize(self):
        est=self.estimator();frame=np.zeros((640,640,3),np.uint8)
        with patch('pose_estimator_yolo26.cv2.resize',wraps=cv2.resize) as resize:
            est.get_keypoints(frame)
        self.assertEqual(resize.call_count,0)

    def test_model_pixels_are_one_resize_of_source_crop(self):
        manager=DualViewManager();frame=np.random.default_rng(3).integers(0,256,(1440,2560,3),dtype=np.uint8)
        dual=manager.split_frame(frame,for_inference=True)
        est=self.estimator()
        for image in (dual.front_frame,dual.back_frame):
            expected=cv2.cvtColor(cv2.resize(image,(640,640)),cv2.COLOR_BGR2RGB)
            np.testing.assert_array_equal(np.asarray(est._preprocess_frame(image)),expected)

    def test_display_and_mask_keep_existing_pixels_and_geometry(self):
        manager=DualViewManager()
        frame=np.random.default_rng(8).integers(0,256,(1440,2560,3),dtype=np.uint8)
        legacy=manager.split_frame(frame)
        direct=manager.split_frame(frame,for_inference=True)
        self.assertEqual(legacy.front_info,direct.front_info)
        self.assertEqual(legacy.back_info,direct.back_info)
        np.testing.assert_array_equal(manager.render_side_by_side(legacy,draw_labels=False),
                                      manager.render_side_by_side(direct,draw_labels=False))
        manager.mask_polygon_norm=[[.40,.20],[.60,.20],[.60,.40],[.40,.40]]
        legacy=manager.split_frame(frame)
        direct=manager.split_frame(frame,for_inference=True)
        self.assertFalse(direct.back_source_pixels)
        np.testing.assert_array_equal(legacy.back_frame,direct.back_frame)

    def test_canonical_points_map_to_same_original_mirror_pixels(self):
        manager=DualViewManager();frame=np.zeros((1440,2560,3),np.uint8)
        direct=manager.split_frame(frame,for_inference=True);est=self.estimator()
        for image,info in [(direct.front_frame,direct.front_info),(direct.back_frame,direct.back_info)]:
            result=est.get_keypoints_from_source(image,coordinate_size=info.view_size)
            point=result[0]['joint0']
            original=info.map_to_original(*point[:2])
            canonical=info.map_from_original(*original)
            np.testing.assert_allclose(canonical,point[:2],atol=1e-9)

    def test_dual_selector_receives_canonical_dimensions(self):
        manager=DualViewManager()
        dual=manager.split_frame(np.zeros((1440,2560,3),np.uint8),for_inference=True)
        model=self.estimator()
        estimator=DualPoseEstimator(backend="mock",concurrent=False)
        self.addCleanup(estimator.close)
        estimator.backend="coreml"
        estimator.model_front=estimator.model_back=estimator.model=model
        self.assertTrue(estimator.supports_source_pixels)
        with patch.object(estimator,"_select_front_candidate_coreml",
                          side_effect=lambda candidates,h,w,**kwargs:candidates[0]) as choose:
            points=estimator._predict_single_view(dual.front_frame,model=model,
                crop_info=dual.front_info,source_pixels=dual.front_source_pixels)
        self.assertEqual(choose.call_args.args[1:3],(720,540))
        self.assertEqual((points['joint0'].x,points['joint0'].y),(270,360))

    def test_front_only_preserves_mirror_model_pixels(self):
        manager=DualViewManager()
        frame=np.random.default_rng(14).integers(0,256,(1440,2560,3),dtype=np.uint8)
        legacy=manager.split_frame(frame)
        front_only=manager.split_frame(frame,for_inference=True,source_views=("front",))
        self.assertTrue(front_only.front_source_pixels)
        self.assertFalse(front_only.back_source_pixels)
        np.testing.assert_array_equal(legacy.back_frame,front_only.back_frame)

    def test_racket_and_ball_skip_identity_resize_without_pixel_changes(self):
        frame=np.random.default_rng(23).integers(0,256,(640,640,3),dtype=np.uint8)
        expected=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        with patch('racket_pose_detector.cv2.resize',wraps=cv2.resize) as resize:
            image,transform=letterbox(frame,640)
        self.assertEqual(resize.call_count,0)
        np.testing.assert_array_equal(image,expected)
        self.assertEqual(transform['gain'],1)
        detector=YOLO26nUnifiedDetector.__new__(YOLO26nUnifiedDetector)
        detector.original_width=detector.original_height=640
        detector.input_width=detector.input_height=640
        for mode in ('stretch','letterbox'):
            detector.preprocess_mode=mode
            with patch('yolo26n_unified_detector.cv2.resize',wraps=cv2.resize) as resize:
                image=detector._preprocess(frame)
            self.assertEqual(resize.call_count,0)
            np.testing.assert_array_equal(np.asarray(image),expected)


if __name__=='__main__':unittest.main()
