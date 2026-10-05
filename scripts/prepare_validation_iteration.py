"""Prepare stages 2-5 for a source-bound run; never manufacture external truth."""
import argparse
import hashlib
import html
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_annotation_evaluation import evaluate_joint_labels
from coach_rule_contract import automatic_coach_policy


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as file:
        file.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source','manifest','journal','events','calibration','output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--joint-tolerance-px', type=float, required=True)
    p.add_argument('--scale-tolerance-m', type=float, required=True)
    a = p.parse_args()
    paths = {k:Path(getattr(a,k)).resolve() for k in ('source','manifest','journal','events','calibration')}
    hashes = {k:digest(v) for k,v in paths.items()}
    manifest = json.loads(paths['manifest'].read_text())
    session_id = manifest['session']['session_id']
    roles = {x['role']:x for x in manifest['artifacts']}
    for key, role in [('journal','frame_journal'),('events','event_snapshot')]:
        if roles[role]['sha256'] != hashes[key]: raise ValueError('Manifest hash mismatch: '+key)
    cal = json.loads(paths['calibration'].read_text())
    if cal['binding']['kind'] != 'video_sha256' or cal['binding']['source_id'] != hashes['source']:
        raise ValueError('Calibration source mismatch')
    rows = [json.loads(line) for line in paths['journal'].read_text().splitlines()]
    events_doc = json.loads(paths['events'].read_text())
    if events_doc['session']['session_id'] != session_id or any(r['session_id'] != session_id for r in rows):
        raise ValueError('Cross-session inputs')
    if any(r.get('ground_reference',{}).get('calibration_id') != cal['calibration_id'] for r in rows):
        raise ValueError('Calibration does not match frame journal')
    events = events_doc['events']
    ids = {r['frame_id']:r for r in rows}
    # Three contiguous windows support future peak/occlusion evaluation, not only clear poses.
    frames = set()
    for event in events:
        anchor = ids[event['contact_frame']]['source_time']['timestamp_seconds']
        for r in rows:
            t = r['source_time'].get('timestamp_seconds')
            if t is not None and -.56 <= t-anchor <= .24: frames.add(r['frame_id'])
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out/'input_binding.json', {'session_id':session_id,'paths':{k:str(v) for k,v in paths.items()},'sha256':hashes})
    write(out/'calibration_snapshot.json', cal)
    scripts = Path(__file__).resolve().parent
    def run(script, arguments):
        subprocess.run([sys.executable,str(scripts/script),*map(str,arguments)],check=True)
    run('build_scale_reference_board.py',['--source',paths['source'],'--calibration',paths['calibration'],
        '--output',out/'scale_reference','--frame',events[1 if len(events)>1 else 0]['contact_frame']])
    run('evaluate_scale_checks.py',['--source',paths['source'],'--calibration',paths['calibration'],
        '--review',out/'scale_reference/blank_measurements.json','--tolerance-m',a.scale_tolerance_m,
        '--output',out/'scale_evaluation.json'])
    run('build_joint_annotation_board.py',['--source',paths['source'],'--output',out/'independent_joints',
        '--frames',','.join(map(str,sorted(frames))),'--torso-only'])
    predictions = {'schema':'tennis.pose-resolution-audit.v1','source_sha256':hashes['source'],
        'scales':[1.0], 'samples':{str(fid):{'1.0':ids[fid].get('pose_observations',{})} for fid in sorted(frames)},
        'prediction_semantics':'Fresh original-pixel observations from current run, not truth',
        'session_id':session_id,'frame_journal_sha256':hashes['journal']}
    write(out/'predictions.json',predictions)
    labels = json.loads((out/'independent_joints/blank_labels.json').read_text())
    write(out/'joint_evaluation.json',evaluate_joint_labels(labels,predictions,a.joint_tolerance_px))
    run('audit_kinematic_cadence.py',['--source',paths['source'],'--manifest',paths['manifest'],
        '--journal',paths['journal'],'--events',paths['events'],'--output',out/'temporal_baseline'])
    write(out/'coach_reference_draft.json',{'schema':'tennis.independent-coach-reference-draft.v1',
        'source_sha256':hashes['source'],'session_id':session_id,'event_snapshot_sha256':hashes['events'],
        'annotator_id':None,'independent_reference':True,'confirmed':False,'rubric':None,
        'rubric_version':None,'tolerance_score':None,'labels':{},
        'requested_event_ids':[e['event_id'] for e in events],
        'requirements':['Record an independent coach assessment before showing model scores.',
                        'Provide defined rule criteria, score units, visibility/unknown decisions.',
                        'Separate rule-development examples from held-out validation examples.',
                        'Three swings alone do not validate general coaching accuracy.']})
    status={'schema':'tennis.iteration-validation-progress.v1','session_id':session_id,
        'source_sha256':hashes['source'],'all_stages_complete':False,
        'stages':{'2':{'status':'awaiting_physical_measurements','evaluation_tool_ready':True},
                  '3':{'status':'awaiting_independent_labels','frames':len(frames),'requested_points':len(frames)*8},
                  '4':{'status':'baseline_audited_pending_error_benchmark','parameter_optimization_accepted':False},
                  '5':{'status':'awaiting_independent_coach_rubric_and_heldout_labels','policy':automatic_coach_policy()}},
        'evaluation_parameters':{'joint_tolerance_px':a.joint_tolerance_px,'scale_tolerance_m':a.scale_tolerance_m,
            'semantics':'Diagnostic parameters only; user acceptance criteria remain unset.'}}
    write(out/'progress.json',status)
    (out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Court02 第2–5项验收</title>
<style>body{font:18px/1.8 system-ui;max-width:1050px;margin:40px auto;padding:0 20px}a{color:#067}td,th{padding:12px;border-bottom:1px solid #bbb;text-align:left}pre{white-space:pre-wrap}</style>
<h1>Court02 第2–5项验收</h1><p>绑定当前新ROI重跑会话。工程工具就绪不代表独立验证通过；原始历史数据保持原样。</p>
<table><tr><th>项目</th><th>当前状态</th><th>下一输入或查看</th></tr>
<tr><td>2 独立尺度</td><td>待现场实测；误差计算工具就绪</td><td><a href="scale_reference/index.html">实测点录入</a></td></tr>
<tr><td>3 关节点误差</td><td>待盲标；连续触球窗口肩髋标注</td><td><a href="independent_joints/index.html">独立标注</a></td></tr>
<tr><td>4 时序与动力链</td><td>已重放审计；待误差基准决定平滑与遮挡参数</td><td><a href="temporal_baseline/report.html">本轮动力链基准</a></td></tr>
<tr><td>5 教练规则与评分</td><td>待教练独立规则、评分和留出验证集</td><td><a href="coach_reference_draft.json">教练评估模板</a></td></tr></table>
<p>尺度检查只针对贴地参照，不能验证人体高度或真实3D转动。图像平面肩髋方向变化也不能当作真实轴向角速度。</p>
<p>独立标注页面不显示模型预测；无法辨认的关节请明确标记，不要猜测。教练评分在独立验证前保持关闭。</p>
<p><a href="progress.json">进度JSON</a> · <a href="input_binding.json">输入身份</a></p><pre>'''+html.escape(json.dumps(status,ensure_ascii=False,indent=2))+'</pre>')
    print(json.dumps(status,ensure_ascii=False))


if __name__ == '__main__': main()
