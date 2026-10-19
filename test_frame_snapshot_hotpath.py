"""Snapshot refresh must reuse durable JSON rows and retain cadence/ownership."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from realtime_swing_pipeline import RealtimeFrameJournal


class FrameSnapshotHotpathTests(unittest.TestCase):
    def test_refresh_reuses_encoded_rows_and_preserves_all_nested_values(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            journal=RealtimeFrameJournal(str(root/'frames.jsonl'),str(root/'latest.json'),
                                         snapshot_size=3,flush_interval=2,queue_size=1)
            snapshots=[]
            encode=json.dumps
            full_list_encodes=[]
            def track(value,*args,**kwargs):
                if isinstance(value,dict) and 'frames' in value:
                    full_list_encodes.append(value)
                return encode(value,*args,**kwargs)
            write=journal._write_snapshot
            def capture():
                write()
                snapshots.append(json.loads((root/'latest.json').read_text()))
            journal._write_snapshot=capture
            original=[]
            with patch('realtime_swing_pipeline.json.dumps',side_effect=track):
                for fid in range(7):
                    row={'frame_id':fid,'label':'中文 "line"\nnext',
                         'pose':{'points':[[1.5,None,False],{'confidence':.7}],'missing':None}}
                    original.append(json.loads(encode(row)))
                    journal.record(row)
                    row['pose']['points'][0][0]=999
                journal.close()
            durable=[json.loads(line) for line in (root/'frames.jsonl').read_text().splitlines()]
            self.assertEqual(durable,original)
            self.assertEqual([s['summary']['frame_count'] for s in snapshots],[2,4,6,7])
            for snapshot in snapshots:
                n=snapshot['summary']['frame_count']
                self.assertEqual(snapshot['frames'],original[max(0,n-3):n])
                self.assertEqual(snapshot['summary']['latest_frame'],n-1)
            self.assertEqual(full_list_encodes,[], 'Snapshot reserializes entire recent-frame trees')
            self.assertEqual(len(journal._recent),3)

    def test_empty_final_snapshot_is_valid(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            journal=RealtimeFrameJournal(str(root/'frames.jsonl'),str(root/'latest.json'))
            journal.close()
            result=json.loads((root/'latest.json').read_text())
            self.assertEqual(result['frames'],[])
            self.assertEqual(result['summary']['latest_frame'],-1)
