"""Exercise the actual event-card JavaScript with a minimal DOM in Node."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node is required for frontend regression')
class ControlPanelKinematicDisplayTests(unittest.TestCase):
    def render(self, sequence):
        page = Path(__file__).with_name('local_control_panel.html').read_text()
        start = page.find('    function kinematicDisplay(')
        if start < 0:
            start = page.index('    function renderSwingsFeed(')
        code = page[start:page.index('    function handleLogsData(', start)]
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


if __name__ == '__main__':
    unittest.main()
