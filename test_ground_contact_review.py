"""The assisted foot board cannot convert ankle hints into contact truth."""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np


class GroundContactReviewTests(unittest.TestCase):
    def test_source_bound_board_retains_hints_without_contact_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'source.mp4'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 25, (64,64))
            for _ in range(3):
                writer.write(np.zeros((64,64,3), dtype=np.uint8))
            writer.release()
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps({'session':{'ground_calibration_application':{'input_binding':{
                'kind':'video_sha256','source_id':hashlib.sha256(source.read_bytes()).hexdigest()}}}}))
            journal = root/'frames.jsonl'
            journal.write_text(json.dumps({'frame_id':0,'pose_observation_coordinate_space':'original_source_pixels',
                'pose_observations':{'front':{'left_ankle':{'x':0,'y':0,'confidence':.9,
                'observed':True,'source_frame_id':0}}}})+'\n')
            evidence=json.loads(manifest.read_text())
            evidence['artifacts']=[{'role':'frame_journal','sha256':hashlib.sha256(journal.read_bytes()).hexdigest()}]
            manifest.write_text(json.dumps(evidence))
            out=root/'board'
            command=[sys.executable,'scripts/build_ground_contact_review.py','--source',str(source),
                     '--manifest',str(manifest),'--journal',str(journal),'--output',str(out),'--frames','0']
            subprocess.run(command,check=True,capture_output=True)
            data=json.loads((out/'ground_contact_review_template.json').read_text())
            self.assertEqual(data['labels'],{})
            self.assertFalse(data['independent_reference'])
            self.assertEqual(data['ankle_references']['0:front:left_ankle']['x'],0)
            page=(out/'index.html').read_text()
            self.assertIn('全部复核已填写项',page)
            self.assertIn('腾空 / 未接地',page)
            if shutil.which('node'):
                script=re.search(r'<script>(.*?)</script>',page,re.S).group(1)
                subprocess.run(['node','--check'],input=script,text=True,check=True,capture_output=True)
                workflow=script[script.index('// Shared by independent'):]
                harness='const data='+json.dumps(data)+''';const names=['left_contact','right_contact'];
const elements={};const $=id=>elements[id]||={value:'',checked:false,textContent:'',selectedIndex:0};
const document={addEventListener(){}};const localStorage={setItem(){},getItem(){return null}};function draw(){};
'''
                checks='''
const label={visible:true,x:0,y:0,contact_state:'ground_contact_visible'};
const good={...data,labels:{'0:front:left_contact':label}};validateAnnotationDraft(good);
for(const bad of [
 {...good,ankle_references:{}},
 {...good,journal_sha256:'other'},
 {...good,labels:{'0:front:left_contact':{...label,contact_state:'airborne'}}},
 {...good,labels:{'0:front:left_ankle':label}}
]){let rejected=false;try{validateAnnotationDraft(bad)}catch(e){rejected=true}if(!rejected)throw Error('unsafe foot draft');}
'''
                subprocess.run(['node','-e',harness+workflow+checks],check=True,capture_output=True)
            before=(out/'index.html').read_bytes()
            # Swapping a journal under the same video binding must be refused.
            journal.write_text(journal.read_text()+'\n')
            failed=subprocess.run(command,capture_output=True)
            self.assertNotEqual(failed.returncode,0)
            self.assertEqual(before,(out/'index.html').read_bytes())
            manifest.write_text(json.dumps({'session':{}}))
            failed=subprocess.run(command,capture_output=True)
            self.assertNotEqual(failed.returncode,0)
            self.assertEqual(before,(out/'index.html').read_bytes())


if __name__=='__main__':
    unittest.main()
