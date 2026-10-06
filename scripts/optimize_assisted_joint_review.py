"""Receive partial human review and automatically audit remaining candidates."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from assisted_joint_optimization import optimize_remaining
from joint_annotation_evaluation import JOINTS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('history','prelabels','predictions','journal','audit','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();raw=Path(a.history).read_bytes();history=json.loads(raw)
    base=json.loads(Path(a.prelabels).read_text());pred_raw=Path(a.predictions).read_bytes();pred=json.loads(pred_raw)
    journal_raw=Path(a.journal).read_bytes();rows=[json.loads(s) for s in journal_raw.decode().splitlines()]
    if history.get('schema')!='tennis.joint-draft-history.v1' or not history.get('history'):raise ValueError('Review history required')
    latest=history['history'][-1]
    for field in ('schema','source_sha256','prediction_sha256','coordinate_space','frame_index_base','frames','requested_joints','model_suggestions','annotation_mode','independent_reference'):
        if latest.get(field)!=base.get(field):raise ValueError('History source contract mismatch: '+field)
    identity=json.loads(history['identity'])
    if identity[:3]!=[base['schema'],base['source_sha256'],base['prediction_sha256']] or identity[3]!=[[f['frame_id'],f['width'],f['height']] for f in base['frames']] or set(identity[4])!=set(base['requested_joints']):raise ValueError('History identity mismatch')
    if hashlib.sha256(pred_raw).hexdigest()!=base['prediction_sha256'] or hashlib.sha256(journal_raw).hexdigest()!=pred['frame_journal_sha256']:raise ValueError('Input hash mismatch')
    audit=json.loads(Path(a.audit).read_text())
    if audit.get('source_sha256')!=base['source_sha256']:raise ValueError('Audit source mismatch')
    if pred.get('source_sha256')!=base['source_sha256'] or any(row.get('session_id')!=pred.get('session_id') for row in rows):raise ValueError('Prediction/session source mismatch')
    expected=set(base['labels'])
    if set(latest['labels'])!=expected:raise ValueError('Missing/extra review labels')
    for key,point in latest['labels'].items():
        fid,view,joint=key.split(':');frame=next(f for f in base['frames'] if f['frame_id']==int(fid))
        if joint not in JOINTS or type(point.get('visible')) is not bool:raise ValueError('Invalid annotation')
        if point['visible']:
            from observation_policy import finite_number
            x,y=finite_number(point.get('x')),finite_number(point.get('y'))
            if x is None or y is None or not 0<=x<frame['width'] or not 0<=y<frame['height']:raise ValueError('Invalid annotation position')
        elif point.get('x') is not None or point.get('y') is not None:raise ValueError('Unidentifiable joint must have null position')
    result,report=optimize_remaining(latest,rows,audit)
    source_sha=hashlib.sha256(raw).hexdigest();result['review_revision_id']=hashlib.sha256((source_sha+report['policy']).encode()).hexdigest()
    result['received_history_sha256']=source_sha
    report.update(source_sha256=base['source_sha256'],received_history_sha256=source_sha,prediction_sha256=base['prediction_sha256'],human_decisions_preserved_exactly=True)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'received_history.json').write_bytes(raw)
    for name,doc in [('received_latest.json',latest),('optimized_review.json',result),('optimization_report.json',report)]:
        (out/name).write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['counts']))


if __name__=='__main__':main()
