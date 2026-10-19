"""Preserve rendered pixels while avoiding repeated font-file loading."""
import logging
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from font_manager import FontManager


class FontHotpathTests(unittest.TestCase):
    def setUp(self):
        try:
            self.path = ImageFont.truetype('DejaVuSans.ttf', 16).path
        except OSError:
            self.path = '/System/Library/Fonts/Supplemental/Arial.ttf'
        if not Path(self.path).exists():
            self.skipTest('No local test font')
        self.manager = object.__new__(FontManager)
        self.manager.default_font = self.path
        self.manager.logger = logging.getLogger('test-font')
        from font_manager import _cached_truetype
        _cached_truetype.cache_clear()
        self.addCleanup(_cached_truetype.cache_clear)

    def test_repeated_frames_load_font_once(self):
        from PIL import ImageFont
        original = ImageFont.truetype
        frame = np.zeros((60, 160, 3), np.uint8)
        with patch.object(ImageFont, 'truetype', wraps=original) as load:
            for _ in range(8):
                self.manager.put_text_with_font(frame, '证据复核', (2, 3))
        self.assertEqual(load.call_count, 1)

    def test_pixel_identity_against_existing_pillow_contract(self):
        for text, org in (('动作复核', (5, 7)), ('证据\nsecond', (-3, -2)), ('多行\n中文', (130, 65))):
            frame = np.random.default_rng(8).integers(0, 256, (80, 160, 3), dtype=np.uint8)
            before = frame.copy()
            actual = self.manager.put_text_with_font(frame, text, org, .7, (12, 180, 240))
            expected = Image.fromarray(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(expected)
            size = int(12 * .7 * 2)
            font = ImageFont.truetype(self.path, size)
            for i, line in enumerate(text.split('\n')):
                draw.text((org[0], org[1] + i * (size + size * .3)), line, font=font, fill=(240, 180, 12))
            np.testing.assert_array_equal(actual, cv2.cvtColor(np.array(expected), cv2.COLOR_RGB2BGR))
            np.testing.assert_array_equal(frame, before)


if __name__ == '__main__':
    unittest.main()
