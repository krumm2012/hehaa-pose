import gc
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

from realtime_swing_pipeline import RealtimeSwingOutputManager


class CountingEvidence(dict):
    visits=0
    def items(self):
        self.visits+=1
        return super().items()


class LiveReportHistoryCacheTests(unittest.TestCase):
    def manager(self, directory):
        result=RealtimeSwingOutputManager.__new__(RealtimeSwingOutputManager)
        result.output_json=Path(directory)/'events.json'
        result.output_html=Path(directory)/'report.html'
        result._render_live_html=lambda document:'<html>unchanged report</html>'
        return result

    @patch('realtime_swing_pipeline.build_session_quality_dashboard',return_value={})
    def test_refresh_reuses_unchanged_history_and_publishes_replacement_event(self, dashboard):
        with tempfile.TemporaryDirectory() as directory:
            manager=self.manager(directory)
            evidence=CountingEvidence(samples=list(range(5000)))
            event={'event_id':1,'evidence':evidence,'clip_status':'pending'}
            document={'summary':{'latest_frame':1},'events':[event],'frame_trace':[]}
            manager._write_live_outputs(document)
            visits=evidence.visits
            document['summary']['latest_frame']=2
            manager._write_live_outputs(document)
            self.assertEqual(evidence.visits,visits,'Unchanged history was serialized again')
            self.assertEqual(json.loads(manager.output_json.read_text())['summary']['latest_frame'],2)
            updated={**event,'clip_status':'ready','evidence':{'reviewed':True}}
            document['events']=[updated]
            manager._write_live_outputs(document)
            saved=json.loads(manager.output_json.read_text())
            self.assertEqual(saved['events'],[updated])
            self.assertEqual(event['clip_status'],'pending')

    @patch('realtime_swing_pipeline.build_session_quality_dashboard',return_value={})
    def test_removed_history_is_not_retained_by_serialization_cache(self, dashboard):
        with tempfile.TemporaryDirectory() as directory:
            manager=self.manager(directory)
            event=CountingEvidence(event_id=1,evidence={'value':'旧证据'})
            ref=weakref.ref(event)
            document={'summary':{},'events':[event]}
            manager._write_live_outputs(document)
            dashboard.reset_mock()  # The observer must not keep the old event alive.
            document['events']=[]
            manager._write_live_outputs(document)
            del event
            gc.collect()
            self.assertIsNone(ref())
            self.assertEqual(json.loads(manager.output_json.read_text())['events'],[])


if __name__=='__main__':unittest.main()
