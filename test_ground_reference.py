import copy
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ground_reference import (GroundReference, SCHEMA, normalize_calibration,
                              map_point, event_ground_reference, ground_reference_html)
from ground_calibration_store import GroundCalibrationStore, source_binding


def calibration():
    return {'schema': SCHEMA, 'image_size': [600, 540], 'width_m': 3., 'length_m': 4.8,
            'dimensions_measured': True, 'camera_geometry_confirmed': True,
            'correspondence_confirmed': True,
            'binding': {'kind': 'video_sha256', 'stream_id': 'court02-main', 'source_id': 'a'*64},
            'views': {'front': {'corner_ids': ['A','B','C','D'],
                               'points': [[10,20],[310,20],[310,500],[10,500]]},
                      'back': {'corner_ids': ['A','B','C','D'],
                               'points': [[500,500],[200,500],[200,20],[500,20]]}}}


def observation(x,y):
    return {'x': x, 'y': y, 'confidence': .99, 'observed': True,
            'source_frame_id': 0, 'confidence_source': 'model', 'recovered_from_mirror': False}


def record():
    return {'frame_id': 0, 'pose_observation_coordinate_space': 'original_source_pixels',
            'pose_observations': {'front': {'left_ankle': observation(60,120),
                                            'right_ankle': observation(160,220)},
                                  'back': {'left_ankle': observation(450,400),
                                           'right_ankle': observation(350,300)}}}


class GroundReferenceTests(unittest.TestCase):
    def engine(self, document=None, binding=None, size=None):
        document=document or calibration()
        return GroundReference(document,binding or document['binding'],size or document['image_size'])

    def test_mirrored_world_correspondences_zero_frame_and_raw_preservation(self):
        raw=record(); old=copy.deepcopy(raw)
        result=self.engine().observe(raw)
        self.assertEqual(raw,old)
        self.assertEqual(result['source_frame_id'],0)
        for foot in ['left_ankle','right_ankle']:
            self.assertAlmostEqual(result['cross_view'][foot]['projection_difference_m'],0)
            self.assertIsNone(result['views']['front'][foot]['position_m'])
        np.testing.assert_allclose(result['views']['front']['left_ankle']['projected_xy_m'],[.5,1.])
        self.assertFalse(result['accuracy_validated'])
        self.assertFalse(result['ground_contact_verified'])
        self.assertFalse(result['coaching_eligible'])

    def test_crop_flip_resize_roundtrip_before_ground_mapping(self):
        from dual_view_manager import DualViewCropInfo
        info=DualViewCropInfo((190,0,590,540),(400,540),(200,270),True)
        local=info.map_from_original(450,400)
        original=info.map_to_original(*local)
        raw=record();raw['pose_observations']['back']['left_ankle']=observation(*original)
        self.assertAlmostEqual(self.engine().observe(raw)['cross_view']['left_ankle']['projection_difference_m'],0)

    def test_grid_renderer_maps_original_corners_and_rejects_stale_context(self):
        from dual_view_manager import DualViewFrame, DualViewCropInfo
        from dual_pose_estimator import DualPoseEstimator
        from dual_view_renderer import DualViewRenderer
        source=np.zeros((540,600,3),dtype=np.uint8)
        front=DualViewCropInfo((0,0,600,540),(600,540),(600,540),False)
        back=DualViewCropInfo((0,0,600,540),(600,540),(600,540),True)
        dual=DualViewFrame(0,source,source.copy(),source.copy(),front,back)
        pose=DualPoseEstimator(backend='mock').estimate_dual_pose(dual)
        renderer=DualViewRenderer(show_hud=False,show_skeleton=False,mask_backview_eyes=False)
        engine=self.engine();reference=engine.observe(record())
        baseline=renderer.render_dual_frame(dual,pose)
        result=renderer.render_dual_frame(dual,pose,ground_reference=reference,ground_geometry=engine.overlay_geometry)
        # A is (10,20) in front, and x=599-500 after the back-view flip.
        self.assertFalse(np.array_equal(result[20,10],baseline[20,10]))
        self.assertFalse(np.array_equal(result[500,600+99],baseline[500,600+99]))
        for change in [{'source_frame_id':1},{'image_size':[1200,1080]},{'reasons':['camera_geometry_not_confirmed']}]:
            actual=renderer.render_dual_frame(dual,pose,ground_reference={**reference,**change},ground_geometry=engine.overlay_geometry)
            np.testing.assert_array_equal(actual,baseline)
        self.assertFalse(source.any())

    def test_unverified_coordinate_system_cannot_enter_mapping(self):
        for space in [None,'roi_pixels','fused_pose']:
            raw=record();raw['pose_observation_coordinate_space']=space
            result=self.engine().observe(raw)
            self.assertEqual(result['views'],{})
            self.assertIn('observation_coordinate_space_unverified',result['reasons'])

    def test_binding_and_size_mismatch_abstain(self):
        for engine,reason in [(self.engine(binding={**calibration()['binding'],'source_id':'b'*64}), 'source_binding_mismatch'),
                              (self.engine(size=[1200,1080]),'source_image_size_mismatch')]:
            result=engine.observe(record())
            self.assertEqual(result['views'],{})
            self.assertIn(reason,result['reasons'])

    def test_unconfirmed_camera_geometry_abstains(self):
        doc=calibration();doc['camera_geometry_confirmed']=False
        self.assertIn('camera_geometry_not_confirmed',self.engine(doc).observe(record())['reasons'])

    def test_unmeasured_dimensions_never_become_measured(self):
        doc=calibration();doc['dimensions_measured']=False
        result=self.engine(doc).observe(record())
        self.assertFalse(result['dimensions_measured'])
        self.assertIn('dimensions_not_measured',result['views']['front']['left_ankle']['reasons'])
        self.assertFalse(result['coaching_eligible'])

    def test_unconfirmed_corner_pairing_does_not_produce_difference(self):
        doc=calibration();doc['correspondence_confirmed']=False
        result=self.engine(doc).observe(record())
        self.assertIsNone(result['cross_view']['left_ankle']['projection_difference_m'])
        self.assertIn('corner_correspondence_not_confirmed',result['cross_view']['left_ankle']['reasons'])

    def test_bad_or_old_points_cannot_be_promoted_by_high_model_score(self):
        changes=[{'x':math.nan},{'y':math.inf},{'confidence':1.2},{'confidence':.4},
                 {'confidence':True},{'observed':False},{'source_frame_id':None},
                 {'source_frame_id':1},{'source_frame_id':False},{'source_frame_id':0.0},
                 {'recovered_from_mirror':True},{'x':400}, {'x':-1}]
        for change in changes:
            with self.subTest(change=change):
                raw=record();raw['pose_observations']['front']['left_ankle'].update(change)
                result=self.engine().observe(raw)
                self.assertIsNone(result['views']['front']['left_ankle']['projected_xy_m'])
                self.assertIsNone(result['cross_view']['left_ankle']['projection_difference_m'])

    def test_legacy_fused_pose_is_not_used_as_a_missing_view(self):
        raw=record();del raw['pose_observations']['back'];raw['pose']={'left_ankle':[450,400]}
        self.assertIsNone(self.engine().observe(raw)['cross_view']['left_ankle']['projection_difference_m'])

    def test_homography_is_recomputed_not_imported(self):
        doc=calibration();doc['views']['front']['H']=[[1,0,0],[0,1,0],[0,0,1]];doc['basis']=[[0,0,0]]
        clean=normalize_calibration(doc)
        self.assertNotIn('basis',clean)
        np.testing.assert_allclose(map_point(clean['views']['front']['H'],[60,120]),[.5,1])
        self.assertEqual(clean,normalize_calibration(clean))

    def test_invalid_geometry_dimensions_flags_and_order_rejected(self):
        cases=[]
        for value in [0,-1,True,'3.3',math.nan,math.inf,1e-300,1e308]:
            doc=calibration();doc['width_m']=value;cases.append(doc)
        for points in [[[10,20]]*4,[[10,20],[310,500],[310,20],[10,500]],
                       [[0,0],[1,1],[2,2],[3,3]],[[10,20],[610,20],[310,500],[10,500]]]:
            doc=calibration();doc['views']['front']['points']=points;cases.append(doc)
        doc=calibration();doc['dimensions_measured']='false';cases.append(doc)
        doc=calibration();doc['views']['back']['corner_ids']=['A','D','C','B'];cases.append(doc)
        for doc in cases:
            with self.subTest(doc=doc), self.assertRaises(ValueError):normalize_calibration(doc)

    def test_extra_ground_point_residual_not_four_corner_accuracy(self):
        doc=calibration();doc['check_points']=[{'view':'front','image':[60,120],'world_m':[.7,1]}]
        clean=normalize_calibration(doc)
        self.assertAlmostEqual(clean['check_points'][0]['error_m'],.2)
        self.assertFalse(clean['accuracy_validated'])
        doc['check_points']*=2
        with self.assertRaises(ValueError):normalize_calibration(doc)
        doc['check_points']=doc['check_points'][:1]
        doc['check_points'][0]['image']=[10,20]
        with self.assertRaises(ValueError):normalize_calibration(doc)

    def test_event_window_samples_null_zero_and_no_version_mixing(self):
        raw=record();raw['ground_reference']=self.engine().observe(raw)
        outside=copy.deepcopy(raw);outside['frame_id']=2;outside['ground_reference']['source_frame_id']=2
        result=event_ground_reference({'start_frame':0,'end_frame':0},[raw,outside])
        self.assertEqual(result['frame_count'],1)
        self.assertAlmostEqual(result['feet']['left_ankle']['median_projection_difference_m'],0)
        self.assertIn('着地未确认',ground_reference_html({'ground_reference':result}))
        duplicate=event_ground_reference({'start_frame':0,'end_frame':0},[raw,raw])
        self.assertIn('duplicate_source_frames',duplicate['rejected_reasons'])
        outside['ground_reference']['calibration_id']='b'*64
        mixed=event_ground_reference({'start_frame':0,'end_frame':2},[raw,outside])
        self.assertIn('mixed_calibration_versions',mixed['rejected_reasons'])
        self.assertEqual(mixed['feet'],{})

    def test_render_actual_live_and_standalone_reports(self):
        from realtime_swing_pipeline import RealtimeSwingOutputManager
        from swing_report_builder import build_report_payload, render_report_html
        raw=record();raw['ground_reference']=self.engine().observe(raw)
        event={'event_id':0,'stroke_type':'Forehand','start_frame':0,'contact_frame':0,'peak_frame':0,'end_frame':0,
               'ground_reference':event_ground_reference({'start_frame':0,'end_frame':0},[raw])}
        manager=RealtimeSwingOutputManager.__new__(RealtimeSwingOutputManager)
        manager.output_json=Path('/tmp/test_ground_events.json');manager.output_html=Path('/tmp/test_ground_report.html')
        manager.roi_metadata={};manager.preview_path=None
        self.assertIn('脚踝地面投影参考',manager._render_live_html({'events':[event],'summary':{}}))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);frames=root/'frames.json';events=root/'events.json'
            frames.write_text(json.dumps({'frames':[]}));events.write_text(json.dumps({'events':[event]}))
            payload=build_report_payload(str(frames),str(events),None)
            page=render_report_html(payload,str(root/'report.html'))
            self.assertIn('脚踝地面投影参考',page)
            self.assertIn('不参与评分',page)

    def test_camera_credential_rotation_does_not_change_binding(self):
        a=source_binding('rtsp://user:first@localhost:8554/lane2?channel=2','court02-main')
        b=source_binding('rtsp://user:second@localhost:8554/lane2?channel=2','court02-main')
        self.assertEqual(a,b)
        self.assertNotIn('user',json.dumps(a))
        self.assertNotEqual(a,source_binding('rtsp://localhost:8554/lane1','court01-main'))
        self.assertNotEqual(a,source_binding('rtsp://localhost:8554/lane2?channel=1','court02-main'))

    def test_immutable_revisions_and_failed_save_preserve_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);video=root/'input.mp4';video.write_bytes(b'actual input bytes')
            binding=source_binding(str(video),'court02-main');doc=calibration();doc['binding']=binding
            store=GroundCalibrationStore(root);first=store.save(doc,binding)
            self.assertEqual(store.save(first,binding),first)
            doc['width_m']=3.3;second=store.save(doc,binding)
            self.assertNotEqual(first['calibration_id'],second['calibration_id'])
            self.assertTrue((store.directory/(first['calibration_id']+'.json')).is_file())
            before={p.name:p.read_bytes() for p in store.directory.iterdir()}
            doc['views']['front']['points']=[[0,0]]*4
            with self.assertRaises(ValueError):store.save(doc,binding)
            self.assertEqual(before,{p.name:p.read_bytes() for p in store.directory.iterdir()})
            self.assertEqual(store.load(binding),second)
            video.write_bytes(b'other input bytes')
            self.assertIsNone(store.load(source_binding(str(video),'court02-main')))


class GroundControlTests(unittest.TestCase):
    def test_actual_export_button_downloads_json_without_shadowing_dom_document(self):
        page=Path('ground_calibration.html').read_text()
        function=page[page.index('function exportDocument('):page.index('function inputChanged(')]
        handler=next(line for line in page.splitlines() if line.startswith("$('export').onclick="))
        doc=calibration()
        setup=f"const source={json.dumps(doc)};"+'''
const context={image_size:source.image_size,binding:source.binding}, origin={kind:'manual_points'};
const draft={front:source.views.front.points,back:source.views.back.points};
const elements={width:{value:'3.3'},length:{value:'4.8'},measured:{checked:true},
 geometry:{checked:true},paired:{checked:false},checks:{value:'[]'},export:{},result:{textContent:''}};
const $=id=>elements[id];
let blob=null,anchor=null,clicks=0,revoked=[];
const document={createElement(tag){if(tag!=='a')throw Error('wrong download element');
 anchor={click(){clicks++}};return anchor;}};
const URL={createObjectURL(value){blob=value;return 'blob:ground-export';},revokeObjectURL(value){revoked.push(value)}};
const setTimeout=callback=>callback();
'''
        assertions='''
(async()=>{
 for(const [geometry,paired] of [[true,false],[false,true],[true,true]]){
  elements.geometry.checked=geometry;elements.paired.checked=paired;
  const before=clicks;elements.export.onclick();
  if(clicks!==before+1)throw Error(elements.result.textContent||'download was not triggered');
  const value=JSON.parse(await blob.text());
  if(anchor.download!=='analyzer_ground_calibration.json'||anchor.href!=='blob:ground-export')throw Error('wrong download');
  if(blob.type!=='application/json'||value.width_m!==3.3||value.length_m!==4.8)throw Error('dimensions or format lost');
  if(value.camera_geometry_confirmed!==geometry||value.correspondence_confirmed!==paired)throw Error('confirmation flags lost');
  if(JSON.stringify(value.views.front.points)!==JSON.stringify(draft.front)||JSON.stringify(value.views.back.points)!==JSON.stringify(draft.back))throw Error('corners lost');
  if(JSON.stringify(value.binding)!==JSON.stringify(context.binding))throw Error('source binding lost');
 }
 if(revoked.length!==3)throw Error('download URLs not released');
 elements.width.value='';const before=clicks;elements.export.onclick();
 if(clicks!==before||!elements.result.textContent.includes('正数宽长'))throw Error('invalid draft downloaded');
})().catch(e=>{console.error(e);process.exitCode=1});
'''
        result=subprocess.run(['node','-e',setup+function+handler+assertions],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_actual_http_token_rejection_and_bad_save_keep_revision(self):
        import threading
        import urllib.request
        import urllib.error
        from http.server import ThreadingHTTPServer
        from local_control_panel import create_handler
        from test_local_control_panel import LocalControlPanelTests
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);controller=LocalControlPanelTests().make_controller(root)
            server=ThreadingHTTPServer(('127.0.0.1',0),create_handler(controller))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base='http://127.0.0.1:'+str(server.server_port)
            def post(path,payload,token=None):
                request=urllib.request.Request(base+path,data=json.dumps(payload).encode(),
                    headers={'Content-Type':'application/json','X-Control-Token':token or controller.token})
                try:
                    with urllib.request.urlopen(request,timeout=3) as response:return response.status,json.load(response)
                except urllib.error.HTTPError as response:return response.code,json.load(response)
            try:
                payload={'stream_id':'court01-main'}
                self.assertEqual(post('/api/ground/preview',payload,token='wrong')[0],403)
                with patch('local_control_panel.capture_calibration_frame',return_value=np.zeros((540,600,3),np.uint8)):
                    status,preview=post('/api/ground/preview',payload)
                self.assertEqual(status,200)
                doc=calibration();doc['binding']=preview['binding']
                request={**payload,'context_id':preview['context_id'],'calibration':doc}
                self.assertEqual(post('/api/ground/save',request)[0],200)
                before=controller.ground_info(payload)['calibration']
                doc['binding']['source_id']='b'*64
                self.assertEqual(post('/api/ground/save',request)[0],400)
                self.assertEqual(controller.ground_info(payload)['calibration'],before)
                self.assertEqual(post('/api/ground/save',{**request,'context_id':'unknown'})[0],400)
            finally:
                controller.stop_server();server.shutdown();server.server_close();thread.join(timeout=3)

    def test_start_loads_calibration_without_touching_camera_config(self):
        import time
        import yaml
        from test_local_control_panel import LocalControlPanelTests
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);controller=LocalControlPanelTests().make_controller(root)
            original=controller.roi_config_path.read_bytes()
            payload={'stream_id':'court01-main','output_dir':'outputs','session_name':'ground',
                     'realtime_swing_events':False,'realtime_coach':False}
            _,binding=controller._ground_binding(payload)
            doc=calibration();doc['binding']=binding
            saved=GroundCalibrationStore(root).save(doc,binding)
            (root/'main_pipe.py').write_text('import time; time.sleep(.2)\n')
            try:
                controller.start(payload)
                runtime=yaml.safe_load((controller._runtime_directory/'runtime_config.yaml').read_text())
                self.assertEqual(runtime['ground_reference']['calibration'],saved)
                self.assertEqual(controller.roi_config_path.read_bytes(),original)
                deadline=time.monotonic()+3
                while controller.status()['state'] in ('running','stopping') and time.monotonic()<deadline:time.sleep(.02)
                self.assertEqual(controller.status()['returncode'],0)
            finally:
                controller.stop()

    def test_real_controller_capture_save_binding_and_protected_configs(self):
        from test_local_control_panel import LocalControlPanelTests
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);controller=LocalControlPanelTests().make_controller(root)
            original=controller.roi_config_path.read_bytes()
            payload={'stream_id':'court01-main'}
            with patch('local_control_panel.capture_calibration_frame',return_value=np.zeros((540,600,3),np.uint8)):
                preview=controller.ground_preview(payload)
            doc=calibration();doc['binding']=preview['binding']
            saved=controller.save_ground_calibration({**payload,'context_id':preview['context_id'],'calibration':doc})
            self.assertEqual(controller.ground_info(payload)['calibration'],saved['calibration'])
            self.assertEqual(controller.roi_config_path.read_bytes(),original)
            doc['image_size']=[1200,1080]
            with self.assertRaises(ValueError):controller.save_ground_calibration({**payload,'context_id':preview['context_id'],'calibration':doc})
            self.assertEqual(controller.ground_info(payload)['calibration'],saved['calibration'])
            with self.assertRaises(ValueError):controller.save_ground_calibration({**payload,'context_id':'missing','calibration':calibration()})

    def test_editor_reference_preflight_and_no_implicit_resolution_conversion(self):
        page=Path('ground_calibration.html').read_text()
        start=page.index('function referencePlan(');end=page.index('function importReference(')
        function=page[start:end]
        context={'image_size':[2560,1440]}
        reference=json.loads(Path('validation/lane2_ground_reference_20261005.json').read_text())
        script=function+f'\nconst context={json.dumps(context)}, reference={json.dumps(reference)};'+'''
let refused=false;try{referencePlan(reference,context,false)}catch(e){refused=true}
if(!refused)throw Error('implicit rescale');
const plan=referencePlan(reference,context,true);
if(plan.draft.front[0][0]!==reference.ground.points[0][0]*2)throw Error('original pixel scale');
if(!plan.measured)throw Error('user confirmed dimensions lost');
const withChecks={...reference,check_points:[{view:'front',image:[800,400],world_m:[1,2]}]};
const preserved=referencePlan(withChecks,context,true).checks[0];
if(preserved.image[0]!==1600||preserved.world_m[0]!==1)throw Error('independent check coordinates lost');
let badCheck=false;try{referencePlan({...withChecks,check_points:[{view:'other',image:[1,2],world_m:[1,2]}]},context,true)}catch(e){badCheck=true}
if(!badCheck)throw Error('bad check accepted');
for(const bad of [{...reference,image_size:[1280,800]}, {...reference,ground:{points:[[true,0],[0,1],[1,1],[1,0]]}}]) {
 let refused=false;try{referencePlan(bad,context,true)}catch(e){refused=true}
 if(!refused)throw Error('invalid reference accepted');
}
'''
        subprocess.run(['node','-e',script],check=True,capture_output=True)


if __name__=='__main__':unittest.main()
