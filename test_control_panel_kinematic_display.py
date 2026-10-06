"""Exercise the actual event-card JavaScript with a minimal DOM in Node."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node is required for frontend regression')
class ControlPanelKinematicDisplayTests(unittest.TestCase):
    def render(self, sequence):
        static_feed = Path(__file__).parent / 'static' / 'js' / 'swing_feed.js'
        page = static_feed.read_text(encoding='utf-8') if static_feed.exists() else Path(__file__).with_name('local_control_panel.html').read_text()
        start = page.find('    function kinematicDisplay(')
        if start < 0:
            start = page.find('function kinematicDisplay(')
        if start < 0:
            start = page.index('function renderSwingsFeed(')
        if '    function handleLogsData(' in page:
            end = page.index('    function handleLogsData(', start)
            code = page[start:end]
        elif 'function handleLogsData(' in page:
            end = page.index('function handleLogsData(', start)
            code = page[start:end]
        else:
            code = page[start:]
        event = {'event_id': 3, 'extended_biomechanics': {'kinematic_sequence': sequence}}
        harness = '''
const elements = {};
const $ = id => elements[id] || (elements[id] = {innerHTML:'', removeAttribute:()=>{}});
const filterOnlyValidSwings = false;
const activeVideoPlayers = new Set();
const isShadowEvent = () => false;
const generateRadarSvg = () => '';
const escapeHtml = s => String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
'''
        result = subprocess.run(['node', '-e', harness + code + '\nrenderSwingsFeed(' + json.dumps([event]) + ");process.stdout.write(elements['swings-feed-list'].innerHTML);"],
                                capture_output=True, text=True, check=True)
        return result.stdout

    def test_unresolved_zero_interval_is_neutral_and_explained(self):
        page = self.render({'sequence_quality': 'UNRESOLVED_AT_FRAME_RATE',
                            'latency_hip_to_shoulder_ms': 0, 'latency_shoulder_to_racket_ms': 0,
                            'coach_eligible': False, 'confidence': 0,
                            'sampling_interval_ms': 41, 'peak_time_uncertainty_ms': 82,
                            'cross_validation': {'status': 'single_view'}})
        self.assertNotIn('UNRESOLVED_AT_FRAME_RATE', page)
        self.assertNotIn('badge-seq disconnected', page)
        self.assertIn('先后难以分辨', page)
        self.assertIn('髋—肩', page)
        self.assertIn('肩—拍', page)

    def test_missing_peaks_are_not_shown_as_zero(self):
        page = self.render({'sequence_quality': None, 'latency_hip_to_shoulder_ms': None,
                            'latency_shoulder_to_racket_ms': None,
                            'cross_validation': {'status': 'unavailable'}})
        self.assertIn('证据不足', page)
        self.assertIn('暂无可用峰值间隔', page)
        self.assertNotIn('0.0 ms', page)
        self.assertNotIn('badge-seq disconnected', page)

    def test_unvalidated_order_never_gets_a_technical_fault_badge(self):
        page = self.render({'sequence_quality': 'DISCONNECTED', 'coach_eligible': False,
                            'confidence': 0, 'latency_hip_to_shoulder_ms': -40,
                            'cross_validation': {'status': 'single_view'}})
        self.assertIn('单视角参考 · 未验证', page)
        self.assertIn('髋—肩 -40.0 ms', page)
        self.assertNotIn('badge-seq disconnected', page)

    def test_candidate_racket_peak_is_shown_as_diagnostic_reference(self):
        page = self.render({'sequence_quality': 'UNRESOLVED_AT_FRAME_RATE',
                            'latency_hip_to_shoulder_ms': 0,
                            'racket_peak_frame': None,
                            'racket_candidate_peak_frame': 25,
                            'racket_candidate_peak_speed': 2304.2,
                            'candidate_latency_shoulder_to_racket_ms': 278.4,
                            'coach_eligible': False, 'confidence': 0,
                            'sampling_interval_ms': 40, 'peak_time_uncertainty_ms': 80,
                            'cross_validation': {'status': 'single_view'}})
        self.assertIn('含候选拍峰', page)
        self.assertIn('肩—拍(候选F25) +278.4 ms @ 2304px/s', page)
    def test_cadence_sensitive_shows_candidates_and_amber_badge(self):
        page = self.render({'sequence_quality': None,
                            'latency_hip_to_shoulder_ms': None,
                            'candidate_hip_peak_frame': 103,
                            'candidate_shoulder_peak_frame': 103,
                            'candidate_latency_hip_to_shoulder_ms': 0.0,
                            'racket_peak_frame': None,
                            'racket_candidate_peak_frame': 102,
                            'racket_candidate_peak_speed': 3180.5,
                            'candidate_latency_shoulder_to_racket_ms': -14.8,
                            'coach_eligible': False, 'confidence': 0,
                            'sampling_interval_ms': 41.3, 'peak_time_uncertainty_ms': 82.7,
                            'cross_validation': {'status': 'unavailable', 'reason': 'cadence_sensitive_peak'}})
        self.assertIn('短时间间隔敏感 · 暂停判定', page)
        self.assertIn('badge-seq cadence_sensitive', page)
        self.assertIn('髋—肩(候选F103) 0.0 ms', page)
        self.assertIn('肩—拍(候选F102) -14.8 ms @ 3181px/s', page)


if __name__ == '__main__':
    unittest.main()
