"""Preserve exact model pixels while bounding RGB conversion to model size."""
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from yolo26n_unified_detector import YOLO26nUnifiedDetector


class UnifiedPreprocessTests(unittest.TestCase):
    def detector(self, frame, mode):
        model = object.__new__(YOLO26nUnifiedDetector)
        model.original_height, model.original_width = frame.shape[:2]
        model.input_width = model.input_height = 960
        model.preprocess_mode = mode
        return model

    def test_rgb_conversion_happens_after_downsampling(self):
        frame = np.zeros((1440, 2560, 3), np.uint8)
        model = self.detector(frame, 'letterbox')
        convert = cv2.cvtColor
        sizes = []
        def capture(image, *args, **kwargs):
            sizes.append(image.shape[:2])
            return convert(image, *args, **kwargs)
        with patch.object(cv2, 'cvtColor', side_effect=capture):
            model._preprocess(frame)
        self.assertTrue(sizes and all(max(size) <= 960 for size in sizes), sizes)

    def test_exact_pixels_match_previous_preprocessing(self):
        rng = np.random.default_rng(13)
        for height, width in ((1440, 2560), (321, 845), (17, 31)):
            frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            before = frame.copy()
            for mode in ('letterbox', 'stretch'):
                model = self.detector(frame, mode)
                if mode == 'letterbox':
                    gain = min(960 / width, 960 / height)
                    nw, nh = max(1, round(width * gain)), max(1, round(height * gain))
                    x, y = (960 - nw) // 2, (960 - nh) // 2
                    expected = np.full((960, 960, 3), 114, np.uint8)
                    expected[y:y + nh, x:x + nw] = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (nw, nh))
                else:
                    expected = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (960, 960))
                np.testing.assert_array_equal(np.asarray(model._preprocess(frame)), expected)
                np.testing.assert_array_equal(frame, before)


if __name__ == '__main__':
    unittest.main()
