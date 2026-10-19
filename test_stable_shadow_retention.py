import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts.retain_stable_shadow_evidence import prune


class RetentionTests(unittest.TestCase):
    def fixture(self,root,wrong_hash=False):
        case=root/'case';case.mkdir()
        video=b'generated video'
        files={'frames.jsonl':b'{"frame_id":0}\n','output.mp4':video,
            'events.json':b'{"events":[]}',
            'evidence_manifest.json':json.dumps({'artifacts':[{'path':'output.mp4','sha256':
                'bad' if wrong_hash else hashlib.sha256(video).hexdigest(),'external':False}]}).encode()}
        archive=root/'case.tar.gz'
        with tarfile.open(archive,'w:gz') as t:
            for name,data in files.items():
                m=tarfile.TarInfo('case/'+name);m.size=len(data);t.addfile(m,io.BytesIO(data))
        return {'session':str(case),'strict_stable_shadow_gate':True,'encoded_frames':250},files,archive

    def test_success_retains_exact_journal_bytes_and_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report,files,original=self.fixture(root)
            with patch('scripts.retain_stable_shadow_evidence.subprocess.check_output',return_value='{"streams":[{"nb_read_frames":"250"}]}'):
                receipt=prune(report,root)
            self.assertFalse(original.exists())
            with tarfile.open(receipt['retained_archive']) as t:
                self.assertEqual(t.extractfile('case/frames.jsonl').read(),files['frames.jsonl'])
                self.assertNotIn('case/output.mp4',t.getnames())
            self.assertEqual(prune(report,root)['retained_archive_sha256'],receipt['retained_archive_sha256'])

    def test_failed_case_wrong_hash_or_count_keeps_original(self):
        for failure in ('failed','hash','count'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);report,_,original=self.fixture(root,wrong_hash=failure=='hash')
                if failure=='failed':report['strict_stable_shadow_gate']=False
                output=json.dumps({'streams':[{'nb_read_frames':'249' if failure=='count' else '250'}]})
                with patch('scripts.retain_stable_shadow_evidence.subprocess.check_output',return_value=output),self.assertRaises(ValueError):
                    prune(report,root)
                self.assertTrue(original.exists())

    def test_outside_task_directory_rejected(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            report,_,original=self.fixture(Path(a))
            with self.assertRaises(ValueError):prune(report,Path(b))
            self.assertTrue(original.exists())


if __name__=='__main__':unittest.main()
