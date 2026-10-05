import copy
import json
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from ground_calibration_store import GroundCalibrationStore, source_binding
from ground_camera_profiles import GroundCameraProfiles
from ground_reference import GroundReference, event_ground_reference
from local_control_panel import create_handler
from test_ground_reference import calibration, record
import test_local_control_panel as control_fixture


class GroundCameraProfileTests(unittest.TestCase):
    def test_actual_editor_file_and_apply_callbacks_preserve_json_flags(self):
        page=Path('ground_calibration.html').read_text()
        callbacks='\n'.join(line for line in page.splitlines() if line.startswith("$('profile-file').onchange=") or line.startswith("$('profile-apply').onclick="))
        doc=self.document()
        setup=f'const original={json.dumps(doc)};'+'''
let profileCandidate=null, calls=[];
const elements={'profile-file':{files:[{name:'lane2.json',size:1000,text:async()=>JSON.stringify(original)}]},
 'profile-apply':{disabled:true},'profile-result':{textContent:''},'profile-target':{value:'court02-main'}};
const $=id=>elements[id];
const request=async(path,payload)=>{calls.push({path,payload});return {message:'已应用',profile:{profile_id:'a'.repeat(64),source_calibration:payload.calibration}}};
'''
        assertions='''
(async()=>{
 await elements['profile-file'].onchange();
 if(elements['profile-apply'].disabled)throw Error('valid file blocked');
 await elements['profile-apply'].onclick();
 if(calls.length!==1||calls[0].path!=='/api/ground/profile/import'||calls[0].payload.target_stream_id!=='court02-main')throw Error('wrong target');
 if(JSON.stringify(calls[0].payload.calibration)!==JSON.stringify(original))throw Error('import changed JSON');
 if(!elements['profile-result'].textContent.includes('镜中对应待确认'))throw Error('unconfirmed correspondence lost');
 elements['profile-file'].files=[{name:'bad.json',size:10,text:async()=>'{bad'}];
 await elements['profile-file'].onchange();await elements['profile-apply'].onclick();
 if(calls.length!==1||!elements['profile-apply'].disabled||profileCandidate!==null)throw Error('bad file reused old candidate');
})().catch(e=>{console.error(e);process.exitCode=1});
'''
        result=subprocess.run(['node','-e',setup+callbacks+assertions],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def controller(self, root):
        controller=control_fixture.LocalControlPanelTests().make_controller(root)
        controller.streams.append({**controller.streams[0], 'stream_id':'court02-main',
                                   'label':'Court 02', 'source':'rtsp://localhost:8554/court02'})
        return controller

    def document(self):
        doc=calibration();doc['width_m']=3.3;doc['correspondence_confirmed']=False
        return doc

    def test_profile_reuse_across_explicitly_mapped_video_bytes_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            profiles=GroundCameraProfiles(directory);doc=self.document();before=copy.deepcopy(doc)
            camera=source_binding('rtsp://localhost:8554/court02','court02-main')
            profile=profiles.save(doc,camera)
            for digest in ['b'*64,'c'*64]:
                binding={**doc['binding'],'source_id':digest}
                resolved=profiles.resolve(binding,camera)
                self.assertEqual(resolved['calibration']['binding'],binding)
                self.assertEqual(resolved['application']['source_calibration_binding'],doc['binding'])
                self.assertEqual(resolved['application']['profile_id'],profile['profile_id'])
                self.assertFalse(resolved['calibration']['correspondence_confirmed'])
                result=GroundReference(resolved['calibration'],binding,[600,540],resolved['application']).observe(record())
                self.assertIsNotNone(result['views']['front']['left_ankle']['projected_xy_m'])
                self.assertIsNone(result['cross_view']['left_ankle']['projection_difference_m'])
                self.assertFalse(result['ground_contact_verified'])
            self.assertEqual(doc,before)
            self.assertEqual(profiles.load(camera),profile)

    def test_other_camera_unmapped_or_changed_address_does_not_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            profiles=GroundCameraProfiles(directory);doc=self.document()
            camera=source_binding('rtsp://localhost:8554/court02?channel=2','court02-main')
            profiles.save(doc,camera)
            self.assertIsNone(profiles.resolve({**doc['binding'],'stream_id':'local_video'},camera))
            self.assertIsNone(profiles.resolve({**doc['binding'],'stream_id':'court01-main'},camera))
            changed=source_binding('rtsp://localhost:8554/court02?channel=1','court02-main')
            self.assertIsNone(profiles.resolve(changed,changed))
            self.assertIsNone(profiles.resolve(changed,camera))

    def test_profile_resize_and_unconfirmed_geometry_abstain(self):
        with tempfile.TemporaryDirectory() as directory:
            profiles=GroundCameraProfiles(directory);doc=self.document()
            camera=source_binding('rtsp://localhost:8554/court02','court02-main')
            profiles.save(doc,camera);resolved=profiles.resolve(doc['binding'],camera)
            engine=GroundReference(resolved['calibration'],doc['binding'],[1200,1080],resolved['application'])
            self.assertIn('source_image_size_mismatch',engine.observe(record())['reasons'])
            doc['camera_geometry_confirmed']=False;profiles.save(doc,camera)
            resolved=profiles.resolve(doc['binding'],camera)
            self.assertIn('camera_geometry_not_confirmed',GroundReference(resolved['calibration'],doc['binding'],[600,540],resolved['application']).observe(record())['reasons'])

    def test_wrong_target_invalid_file_and_failed_import_preserve_revisions(self):
        with tempfile.TemporaryDirectory() as directory:
            profiles=GroundCameraProfiles(directory);camera=source_binding('rtsp://localhost/court02','court02-main')
            first=profiles.save(self.document(),camera)
            before={p.name:p.read_bytes() for p in profiles.directory.iterdir()}
            for document in [None,{'schema':'wrong'},{**self.document(),'width_m':True},
                             {**self.document(),'binding':{**self.document()['binding'],'stream_id':'court01-main'}}]:
                with self.subTest(document=document),self.assertRaises(ValueError):profiles.save(document,camera)
            self.assertEqual(before,{p.name:p.read_bytes() for p in profiles.directory.iterdir()})
            changed=self.document();changed['width_m']=3.4;second=profiles.save(changed,camera)
            self.assertNotEqual(first['profile_id'],second['profile_id'])
            self.assertTrue((profiles.directory/(first['profile_id']+'.json')).exists())

    def test_actual_controller_import_and_video_and_live_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            controller=self.controller(Path(directory));original=controller.roi_config_path.read_bytes()
            controller.import_ground_camera_profile({'target_stream_id':'court02-main','calibration':self.document()})
            upload=controller.workspace/'data/control_uploads';upload.mkdir(parents=True)
            video_id='b'*32+'.mp4';(upload/video_id).write_bytes(b'separate video bytes')
            payload={'stream_id':'local_video','mapped_stream_id':'court02-main','video_id':video_id}
            info=controller.ground_info(payload)
            self.assertEqual(info['application']['scope'],'camera_profile')
            self.assertEqual(info['calibration']['width_m'],3.3)
            self.assertIsNone(controller.ground_info({**payload,'mapped_stream_id':''})['calibration'])
            self.assertIsNone(controller.ground_info({**payload,'mapped_stream_id':'court01-main'})['calibration'])
            self.assertEqual(controller.ground_info({'stream_id':'court02-main'})['application']['scope'],'camera_profile')
            self.assertEqual(controller.roi_config_path.read_bytes(),original)

    def test_shared_profile_priority_and_preview_resolution_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            controller=self.controller(Path(directory));payload={'stream_id':'court02-main'}
            _,binding=controller._ground_binding(payload)
            old=self.document();old['binding']=binding;old['width_m']=3.1
            GroundCalibrationStore(directory).save(old,binding)
            controller.import_ground_camera_profile({'target_stream_id':'court02-main','calibration':self.document()})
            self.assertEqual(controller.ground_info(payload)['calibration']['width_m'],3.3)
            with patch('local_control_panel.capture_calibration_frame',return_value=np.zeros((1080,1200,3),np.uint8)):
                preview=controller.ground_preview(payload)
            self.assertIsNone(preview['calibration'])
            self.assertIn('尺寸不同',preview['calibration_unavailable_reason'])
            self.assertEqual(GroundCalibrationStore(directory).load(binding)['width_m'],3.1)

    def test_actual_start_runtime_uses_profile_without_capture_or_roi_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);controller=self.controller(root);before=controller.roi_config_path.read_bytes()
            controller.import_ground_camera_profile({'target_stream_id':'court02-main','calibration':self.document()})
            upload=root/'data/control_uploads';upload.mkdir(parents=True);video_id='d'*32+'.mp4';(upload/video_id).write_bytes(b'next video')
            (root/'main_pipe.py').write_text('import time;time.sleep(.1)\n')
            try:
                controller.start({'stream_id':'local_video','mapped_stream_id':'court02-main','video_id':video_id,'output_dir':'outputs','session_name':'profile','realtime_swing_events':False,'realtime_coach':False})
                runtime=yaml.safe_load((controller._runtime_directory/'runtime_config.yaml').read_text())
                self.assertEqual(runtime['ground_reference']['application']['scope'],'camera_profile')
                self.assertEqual(runtime['ground_reference']['calibration']['binding']['source_id'],source_binding(str(upload/video_id),'court02-main')['source_id'])
                self.assertFalse(runtime['ground_reference']['calibration']['correspondence_confirmed'])
                deadline=time.monotonic()+3
                while controller.status()['state'] in ('running','stopping') and time.monotonic()<deadline:time.sleep(.02)
                self.assertEqual(controller.status()['returncode'],0)
                self.assertEqual(before,controller.roi_config_path.read_bytes())
            finally:controller.stop()

    def test_actual_http_import_requires_token_and_rejects_wrong_court(self):
        with tempfile.TemporaryDirectory() as directory:
            controller=self.controller(Path(directory));server=ThreadingHTTPServer(('127.0.0.1',0),create_handler(controller))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            def post(target,token=None):
                req=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/api/ground/profile/import',data=json.dumps({'target_stream_id':target,'calibration':self.document()}).encode(),headers={'Content-Type':'application/json','X-Control-Token':token or 'wrong'},method='POST')
                try:
                    with urllib.request.urlopen(req) as response:return response.status,json.load(response)
                except urllib.error.HTTPError as error:return error.code,json.load(error)
            try:
                self.assertEqual(post('court02-main')[0],403)
                self.assertEqual(post('court02-main',controller.token)[0],200)
                before=controller.ground_info({'stream_id':'court02-main'})
                self.assertEqual(post('court01-main',controller.token)[0],400)
                self.assertEqual(post('unknown',controller.token)[0],400)
                self.assertEqual(controller.ground_info({'stream_id':'court02-main'}),before)
            finally:server.shutdown();server.server_close();thread.join(timeout=3);controller.stop_server()

    def test_frame_and_event_application_identity_and_mixed_profile_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            profiles=GroundCameraProfiles(directory);doc=self.document();camera=source_binding('rtsp://localhost/court02','court02-main')
            profiles.save(doc,camera);resolved=profiles.resolve(doc['binding'],camera)
            engine=GroundReference(resolved['calibration'],doc['binding'],[600,540],resolved['application'])
            frame=record();frame['ground_reference']=engine.observe(frame)
            event=event_ground_reference({'start_frame':0,'end_frame':0},[frame])
            self.assertEqual(event['application'],resolved['application'])
            wrong={**resolved['application'],'input_binding':{**doc['binding'],'source_id':'e'*64}}
            self.assertIn('application_binding_mismatch',GroundReference(resolved['calibration'],doc['binding'],[600,540],wrong).observe(record())['reasons'])
            other=copy.deepcopy(frame);other['frame_id']=1;other['ground_reference']['source_frame_id']=1;other['ground_reference']['application']['profile_id']='f'*64
            mixed=event_ground_reference({'start_frame':0,'end_frame':1},[frame,other])
            self.assertIn('mixed_camera_profile_versions',mixed['rejected_reasons'])


if __name__=='__main__':unittest.main()
