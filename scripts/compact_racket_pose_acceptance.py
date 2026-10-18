"""Keep first-round comparisons, failures, and all records; prune repeat video derivatives.

Only operates on this acceptance runner's generated, measured session archives.
Source media, annotations, models, and unrelated historical results are never read
for deletion. Required replay artifacts are retained with their original bytes.
"""
import argparse
import hashlib
import io
import json
import tarfile
import shutil
from pathlib import Path

from validate_racket_pose_pipeline import compact_repeat


def checksum(stream):
    h=hashlib.sha256()
    for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--prune-experimental-models',action='store_true');parser.add_argument('--group',action='append',default=[]);parser.add_argument('--keep-group',action='append',default=[]);parser.add_argument('--keep-session',action='append',default=[]);args=parser.parse_args()
    root=Path(args.root).resolve()
    for measurement in sorted(root.glob('*/*/measurement.json')):
        report=json.loads(measurement.read_text());session=measurement.parent
        if args.group and session.parent.name not in args.group:continue
        if report.get('returncode')!=0:continue  # Failed/crashed sessions keep their diagnostics intact.
        if Path(report['session']).resolve()!=session:raise ValueError('Session identity mismatch')
        # Keep final nine-clip comparisons and representative failures. All
        # measurements, journals, source media and replay records remain intact.
        failure_examples={
            ('final-single','realtime-active-s1-49.23-r0'),
            ('dual-comparison','realtime-active-s1-48.43-r0'),
            ('dual-comparison','realtime-active-s2-49.23-r1'),
            ('shared-output-smoke-single-v2','realtime-active-s1-50.03-r0'),
        }
        keep=(session.parent.name in args.keep_group or session.name in args.keep_session or session.relative_to(root).as_posix() in args.keep_session
              or session.parent.name=='final-single-s2' and report['repeat']==0
              or session.parent.name=='final-single' and report['mode']=='legacy'
              or session.parent.name=='dual-comparison' and report['mode']=='active' and report['stride']==3 and report['repeat']==0
              or (session.parent.name,session.name) in failure_examples)
        if keep:continue
        if not report.get('verified_archive'):
            compact_repeat(session,report)
        old=Path(report['verified_archive']).resolve()
        if old.parent!=session.parent:raise ValueError('Archive escapes session group')
        if old.name.endswith('.records.tar.gz'):continue
        new=old.with_name(session.name+'.records.tar.gz')
        if new.exists():raise FileExistsError(new)
        retained={};omitted=[]
        with tarfile.open(old,'r:gz') as source,tarfile.open(new,'w:gz',compresslevel=1) as dest:
            manifest=json.load(source.extractfile(session.name+'/evidence_manifest.json'))
            removable={session.name+'/'+r['path'] for r in manifest['artifacts'] if not r['external'] and not r['required_for_replay'] and r['role'] in ('annotated_video','swing_clip')}
            for member in source:
                stream=source.extractfile(member) if member.isfile() else None
                if member.name in removable:
                    omitted.append({'path':member.name,'bytes':member.size,'sha256':checksum(stream),'reason':'Redundant generated video derivative; original media and replay records retained'})
                    continue
                if stream:
                    data=stream.read();retained[member.name]=hashlib.sha256(data).hexdigest();dest.addfile(member,io.BytesIO(data))
                else:dest.addfile(member)
            data=json.dumps({'policy':'Keep source, first-round visuals, failures and all replay records; omit duplicate successful-run video derivatives','omitted':omitted},ensure_ascii=False,indent=2).encode()
            member=tarfile.TarInfo(session.name+'/derivative_retention.json');member.size=len(data);dest.addfile(member,io.BytesIO(data));retained[member.name]=hashlib.sha256(data).hexdigest()
        verified={}
        with tarfile.open(new,'r:gz') as archive:
            for member in archive:
                if member.isfile():verified[member.name]=checksum(archive.extractfile(member))
        if verified!=retained:raise ValueError('Retained record hash verification failed')
        report['verified_archive']=str(new);report['omitted_derived_videos']=omitted
        measurement.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        old.unlink()  # This run's redundant generated archive, only after verification.
        print(session.name,'retained',len(retained),'records; omitted',len(omitted),'video derivatives',flush=True)
    for group in ('soak','soak-after-history-fix','soak-shared-output','soak-final-s2'):
        if args.group and group not in args.group:continue
        session=root/group
        if not (session/'run.log').exists():continue
        videos=sorted((session/'output_swing_clips').glob('*.mp4'))+sorted((session/'output_swing_clips').glob('*.jpg'))
        keep=set()
        for ext in ('.mp4','.jpg'):
            selected=[p for p in videos if p.suffix==ext];keep.update(selected[:2]+selected[-2:])
        omitted=[]
        for video in videos:
            if video in keep:continue
            with video.open('rb') as stream:sha=checksum(stream)
            omitted.append({'path':video.relative_to(session).as_posix(),'bytes':video.stat().st_size,'sha256':sha})
        if omitted:
            inventory=session/'derivative_retention.json'
            if inventory.exists():raise FileExistsError(inventory)
            inventory.write_text(json.dumps({'policy':'Retain all measurements, journals, event history and freeze images; keep first/last clip examples; omit redundant generated soak videos','omitted':omitted},ensure_ascii=False,indent=2))
            for item in omitted:(session/item['path']).unlink()
            print(group,'retained all records; omitted',len(omitted),'soak video derivatives',flush=True)
    if args.prune_experimental_models:
        from racket_pose_detector import model_digest
        candidate=root.parents[2]/'models/racket_pose/candidates/bn-recalibrated-v1'
        names=('bn_recalibrated_fp32.mlpackage','bn_recalibrated_mixed.mlpackage',
               'bn_repaired_bounded_fp16.mlpackage','bn_repaired_conv16.mlpackage',
               'bn_repaired_float32.mlpackage','bn_repaired_mixed.mlpackage','bn_repaired_safe.mlpackage')
        inventory=[]
        for name in names:
            path=candidate/name
            if not path.exists():continue
            if path.is_symlink():raise ValueError('Experimental model must not be a symlink')
            inventory.append({'path':str(path),'sha256':model_digest(path),
                'files':[{'path':p.relative_to(path).as_posix(),'bytes':p.stat().st_size,'sha256':model_digest(p)} for p in path.rglob('*') if p.is_file()]})
        if inventory:
            target=root/'experimental-export-inventory.json'
            if target.exists():raise FileExistsError(target)
            target.write_text(json.dumps({'scope':'Only superseded exports created during this acceptance task. Keep original supplied models, both repair checkpoints, calibration/quality evidence and the validated release.','superseded':inventory},ensure_ascii=False,indent=2))
            for item in inventory:shutil.rmtree(item['path'])
            print('Removed',len(inventory),'superseded experimental export packages; retained hashes and reconstruction sources',flush=True)


if __name__=='__main__':main()
