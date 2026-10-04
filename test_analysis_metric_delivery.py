"""Regression: event measurements reach summaries and replay uses its own event."""
import json
import threading
import tempfile
import unittest
import subprocess
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np

from swing_session_summary import build_session_coaching_summary
from realtime_swing_pipeline import RealtimeSwingOutputManager
from analysis_metric_delivery import event_analysis_metrics, scoring_blockers


class MetricDeliveryTests(unittest.TestCase):
    def test_live_report_uses_pixel_speed_and_qualified_zero_rise(self):
        manager = RealtimeSwingOutputManager.__new__(RealtimeSwingOutputManager)
        manager.output_json=Path('/tmp/current_events.json');manager.output_html=Path('/tmp/current_report.html')
        manager.roi_metadata={};manager.preview_path=None
        event={'event_id':1,'stroke_type':'Forehand','start_frame':0,'end_frame':20,'contact_frame':12,
               'extended_biomechanics':{'racket_head_speed':{'contact_kmh':180,'contact_px_s':250,'max_px_s':500},
                 'brush_angle':{'low_to_high_angle_deg':22,'drop_depth_ratio':0,
                      'measurement_evidence':{'fields':{'low_to_high_angle_deg':{'display_eligible':False},
                                                        'drop_depth_ratio':{'display_eligible':True}}}}}}
        page=manager._render_live_html({'events':[event],'summary':{}})
        self.assertIn('250 px/s',page)
        self.assertIn('上升比 0.00x',page)
        self.assertNotIn('180.0 km/h',page)
        self.assertNotIn('+22.0°',page)

    def test_five_dimension_blockers_are_separate_from_zero_measurements(self):
        event = {'contact_status':'candidate','biomechanics':{'metrics':{
            'contact_lateral_distance':{'value':0},'weight_transfer':{'value':None}}}}
        blocks = {r['dimension']:r for r in scoring_blockers(event)}
        self.assertEqual(len(blocks),5)
        self.assertIn('automatic_rubric_not_independently_validated',blocks['contact']['reasons'])
        self.assertIn('contact_not_confirmed',blocks['contact']['reasons'])
        self.assertNotIn('missing_observations',blocks['contact']['reasons'])
        self.assertIn('weight_transfer',blocks['positioning']['missing_observations'])
        from practice_scoring import DIMENSIONS
        event['practice_review']={'confirmed':True,'ratings':{k:3 for k in DIMENSIONS}}
        self.assertEqual(scoring_blockers(event),[])

    def test_unqualified_video_proxies_are_visible_without_becoming_valid_metrics(self):
        event = {'biomechanics': {'metrics': {'racket_head_speed': {'value': None, 'confidence': 0}}},
                 'extended_biomechanics': {'racket_head_speed': {'contact_px_s': 435},
                                         'brush_angle': {'low_to_high_angle_deg': 0, 'drop_depth_ratio': 0}}}
        rows = {r['key']: r for r in event_analysis_metrics(event)}
        self.assertEqual(rows['racket_head_speed']['value'], 435)
        self.assertFalse(rows['racket_head_speed']['coach_eligible'])
        self.assertEqual(rows['brush_angle']['value'], 0)
        self.assertEqual(rows['drop_depth_ratio']['value'], 0)

    @unittest.skipUnless(shutil.which('node'), 'Node required for frontend regression')
    def test_real_page_renders_metrics_and_refreshes_updates_to_same_event(self):
        page = Path(__file__).with_name('local_control_panel.html').read_text()
        helpers = page[page.index('    function measurementValue('):page.index('    function renderSwingsFeed(')]
        handler = page[page.index('    function handleSwingsData('):page.index('    async function pollSwings(')]
        summary = build_session_coaching_summary([{'event_id': 2, 'stroke_type': 'Backhand',
            'biomechanics': {'metrics': {'arm_extension': {'value': 88.67, 'unit': 'deg_2d'}}}}])
        harness = '''
let latestRawEvents = [], lastEventsJson = '', filterOnlyValidSwings = false, calls = 0;
const escapeHtml = value => String(value).replaceAll('<','&lt;');
const renderSessionSummary = () => calls++;
const renderSwingsFeed = () => calls++;
'''
        script = harness + helpers + handler + '\nconst summary = ' + json.dumps(summary) + ''';
const event = {event_id:2, analysis_metrics:[{label:'手臂伸展角',value:88.67,unit:'deg_2d'}]};
console.log(renderEventMeasurements(event)); console.log(renderMeasurementSummary(summary.analysis_metrics));
handleSwingsData({events:[event],summary});
event.analysis_metrics[0].value = 0;
handleSwingsData({events:[event],summary});
if(calls !== 4) throw new Error('same event measurement update was suppressed');
console.log(renderEventMeasurements(event));
'''
        result = subprocess.run(['node', '-e', script], capture_output=True, text=True, check=True).stdout
        self.assertIn('手臂伸展角', result)
        self.assertIn('88.67 °（二维）', result)
        self.assertIn('0 °（二维）', result)
        self.assertIn('1/1', result)

    def test_summary_keeps_zero_missing_units_and_shadow_separate(self):
        def event(eid, value, unit='deg', shadow=False):
            return {'event_id': eid, 'is_shadow_swing': shadow, 'biomechanics': {'metrics': {
                'brush_angle': {'value': value, 'unit': unit, 'confidence': 0, 'coach_eligible': False}}}}
        summary = build_session_coaching_summary([
            event(1, 0), event(2, 10), event(3, None), event(4, 100, shadow=True), event(5, 2, 'ratio')])
        rows = summary['analysis_metrics']
        angle = next(r for r in rows if r['key'] == 'brush_angle' and r['unit'] == 'deg')
        self.assertEqual(angle['median'], 5)
        self.assertEqual(angle['count'], 2)
        self.assertEqual(angle['total_events'], 4)
        self.assertEqual(angle['coach_eligible_count'], 0)
        self.assertEqual(len(rows), 2)

    def test_clip_and_freeze_replace_cached_previous_event_card(self):
        # The cached pixels contain the previous event's card; only current event may be supplied.
        manager = RealtimeSwingOutputManager.__new__(RealtimeSwingOutputManager)
        manager._buffer_condition = threading.Condition()
        manager._latest_buffered_frame = 88
        manager._compressor_stopped = True
        manager.width, manager.height, manager.fps = 1080, 720, 25
        manager.video_backend, manager.video_bitrate = 'opencv', None
        manager.is_dual_view = True
        old = np.full((720, 1080, 3), 72, np.uint8)
        encoded = cv2.imencode('.jpg', old)[1].tobytes()
        manager._frames_for_event_locked = lambda event: [(88, encoded)]
        event = {'event_id': 2, 'contact_frame': 88, 'stroke_type': 'Backhand',
                 'extended_biomechanics': {'racket_head_speed': {'contact_kmh': 180}}}
        renderer = MagicMock()
        renderer.draw_impact_telemetry_card.side_effect = lambda frame, card, **kw: np.full_like(frame, 180)
        writer = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            manager.output_json = Path(directory) / 'events.json'
            with patch('realtime_swing_pipeline.create_video_writer', return_value=writer), \
                 patch('dual_view_renderer.DualViewRenderer', return_value=renderer):
                result = manager._encode_event_clip(Path(directory) / 'event_0002.mp4', event, 88, [88])
            self.assertGreaterEqual(renderer.draw_impact_telemetry_card.call_count, 2)
            for call in renderer.draw_impact_telemetry_card.call_args_list:
                self.assertEqual(call.args[1]['racket_speed_kmh'], 180)
                self.assertEqual(call.kwargs['opacity'], 1.0)
            self.assertEqual(int(writer.write.call_args.args[0][400, 400, 0]), 180)
            freeze = cv2.imread(str(Path(directory) / result['impact_freeze_path']))
            self.assertEqual(int(freeze[400, 400, 0]), 180)
