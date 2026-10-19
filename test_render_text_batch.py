"""Lock down text pixels/order while bounding color conversion and image copies."""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from test_font_hotpath import FontHotpathTests


class RenderTextBatchTests(FontHotpathTests):
    def reference(self, frame, items):
        canvas = frame.copy()
        for item in items:
            text, org = item['text'], item['org']
            scale, color = item.get('font_scale', .7), item.get('color', (255,255,255))
            if any('\u4e00' <= c <= '\u9fff' for c in text):
                pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil)
                size = int(24 * scale)
                font = ImageFont.truetype(self.path, size)
                for i, line in enumerate(text.split('\n')):
                    draw.text((org[0], org[1]+i*(size+size*.3)), line, font=font, fill=color[::-1])
                canvas = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            else:
                for i,line in enumerate(text.split('\n')):
                    cv2.putText(canvas,line,(org[0],org[1]+i*int(30*scale)),
                        item.get('font_face',cv2.FONT_HERSHEY_SIMPLEX),scale,color,
                        item.get('thickness',2),cv2.LINE_AA)
        return canvas

    def test_chinese_converts_only_ink_region(self):
        frame = np.zeros((720,1080,3),np.uint8)
        converted = []
        original = cv2.cvtColor
        def track(img, code, *args, **kwargs):
            converted.append(img.shape[0]*img.shape[1])
            return original(img,code,*args,**kwargs)
        with patch.object(cv2,'cvtColor',side_effect=track):
            self.manager.put_text_with_font(frame,'动作证据不足',(15,45))
        self.assertTrue(converted)
        self.assertLess(max(converted), frame.shape[0]*frame.shape[1]//10)

    def test_ordered_batch_and_clipped_multiline_match_full_canvas_reference(self):
        frame = np.random.default_rng(26).integers(0,256,(120,240,3),dtype=np.uint8)
        items = [dict(text='ABC\nDEF',org=(5,20),color=(45,91,183)),
                 dict(text='证据ABC\n中文',org=(-4,-3),font_scale=.7,color=(19,80,221)),
                 dict(text='OVERLAP',org=(0,20),font_scale=.4),
                 dict(text='中文\nsecond',org=(210,95),font_scale=.55),
                 dict(text='中文outside',org=(-600,-500)),
                 dict(text=' \n中文',org=(5,100))]
        before = frame.copy()
        actual = self.manager.put_texts_with_font(frame,items)
        np.testing.assert_array_equal(actual,self.reference(frame,items))
        np.testing.assert_array_equal(frame,before)

    def test_fractional_origins_and_scales_preserve_full_canvas_pixels(self):
        frame = np.random.default_rng(11).integers(0,256,(90,190,3),dtype=np.uint8)
        for org in [(-25.5,-3.7),(1.2,2.8),(180,82),(-300,4),(4,100)]:
            for scale in (.4,.55,.7,1.2):
                item=dict(text='  中文 A\nsecond行\n',org=org,font_scale=scale,color=(14,190,71))
                np.testing.assert_array_equal(self.manager.put_text_with_font(frame,**item),self.reference(frame,[item]))

    def test_concurrent_cached_faces_keep_same_pixels(self):
        frame=np.zeros((90,190,3),np.uint8)
        items=[dict(text='中文\n证据复核',org=(3,2))]
        expected=self.reference(frame,items)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(lambda _:self.manager.put_texts_with_font(frame,items),range(12)):
                np.testing.assert_array_equal(result,expected)

    def test_telemetry_card_keeps_full_canvas_reference_pixels(self):
        from dual_view_renderer import DualViewRenderer
        owner = self
        class FullCanvasText:
            def put_texts_with_font(self, frame, items):
                return owner.reference(frame, items)
        actual_renderer, expected_renderer = DualViewRenderer(), DualViewRenderer()
        actual_renderer._font_mgr = self.manager
        expected_renderer._font_mgr = FullCanvasText()
        frame=np.random.default_rng(29).integers(0,256,(720,1080,3),dtype=np.uint8)
        for card in ({}, {'kinematic_sequence_text':'证据不足 · 含候选拍峰','hip_evidence_text':'未观测'},
                     {'swing_score':75.5,'swing_grade':'ADVANCED','racket_speed_kmh':90.2,
                      'racket_speed_px_s':1344,'brush_angle_deg':-14.5,'drop_depth_ratio':.81,
                      'foot_line_angle_deg':70.2,'hip_rise_px':23.1,'leg_drive_ratio':.4}):
            before=frame.copy()
            np.testing.assert_array_equal(actual_renderer.draw_impact_telemetry_card(frame,card),
                                          expected_renderer.draw_impact_telemetry_card(frame,card))
            np.testing.assert_array_equal(frame,before)
