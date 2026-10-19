"""Prune only verified successful-run video derivatives created for this release.

Original media and pre-existing results are outside the allowed task directory.
All journals, images, original manifests, failed-run videos and selected examples
remain. The retention receipt explicitly records omitted generated video assets.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.validate_racket_pose_pipeline import compact_repeat

TASK=ROOT/'data/analysis_results/racket_pose_acceptance_20261018/stable-shadow-release-20261019'


def digest(stream):
    h=hashlib.sha256()
    for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def prune(report, task_root=TASK):
    if report.get('strict_stable_shadow_gate') is not True:
        raise ValueError('Failed or unaccepted runs must retain their videos')
    session=Path(report['session']).resolve()
    session.relative_to(Path(task_root).resolve())
    target=session.parent/(session.name+'.journals.tar.gz')
    receipt_path=session/'video-retention.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if not Path(receipt['retained_archive']).is_file():raise ValueError('Retained archive missing')
        return receipt
    if target.exists():raise FileExistsError(target)
    archive=session.parent/(session.name+'.tar.gz')
    if not archive.exists():compact_repeat(session,report)
    expected={};videos=[]
    with tarfile.open(archive,'r:gz') as source:
        events=json.load(source.extractfile(session.name+'/events.json'))['events']
        manifest=json.load(source.extractfile(session.name+'/evidence_manifest.json'))
        expected_hashes={a['path']:a['sha256'] for a in manifest['artifacts'] if a.get('sha256') and not a.get('external')}
        clip_counts={Path(e['clip_path']).name:e['clip_frame_count'] for e in events}
        with tarfile.open(target,'w:gz',compresslevel=1) as dest, tempfile.TemporaryDirectory() as tmp:
            for member in source:
                if member.issym() or member.islnk():raise ValueError('Unexpected link in regression archive')
                if not member.isfile():dest.addfile(member);continue
                stream=source.extractfile(member)
                if member.name.endswith('.mp4'):
                    path=Path(tmp)/Path(member.name).name
                    with path.open('wb') as out:
                        for b in iter(lambda:stream.read(1024*1024),b''):out.write(b)
                    with path.open('rb') as f:sha=digest(f)
                    value=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
                        '-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','json',str(path)],text=True))
                    count=int(value['streams'][0]['nb_read_frames'])
                    relative=Path(member.name).relative_to(session.name).as_posix()
                    if expected_hashes.get(relative)!=sha:raise ValueError('Video hash differs from original evidence manifest')
                    expected_count=report['encoded_frames'] if path.name=='output.mp4' else clip_counts[path.name]
                    if count!=expected_count:raise ValueError('Encoded count mismatch; keep full original archive')
                    videos.append({'path':member.name,'sha256':sha,'bytes':path.stat().st_size,'decoded_frames':count})
                    path.unlink()
                else:
                    data=stream.read();expected[member.name]=hashlib.sha256(data).hexdigest()
                    dest.addfile(member,io.BytesIO(data))
            receipt={'scope':'Successful derivative videos generated in this task only; originals/history untouched',
                'session':str(session),'video_assets_omitted_after_verification':videos,
                'retained_archive':str(target),'original_archive_sha256':None,
                'original_manifest_note':'Original manifest preserved as generation-time evidence. Omitted video assets are intentionally unavailable after retention, not newly verified as present.'}
            with archive.open('rb') as f:receipt['original_archive_sha256']=digest(f)
            data=(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode()
            name=session.name+'/video-retention.json';member=tarfile.TarInfo(name);member.size=len(data)
            dest.addfile(member,io.BytesIO(data));expected[name]=hashlib.sha256(data).hexdigest()
    seen={}
    with tarfile.open(target,'r:gz') as verify:
        for member in verify:
            if member.isfile():seen[member.name]=digest(verify.extractfile(member))
    if seen!=expected:raise ValueError('Retained evidence archive verification failed; original remains')
    receipt['retained_files_sha256']={name:h for name,h in expected.items() if not name.endswith('/video-retention.json')}
    with target.open('rb') as f:receipt['retained_archive_sha256']=digest(f)
    receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    archive.unlink()
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true',help='Otherwise only lists eligible new successful runs')
    args=parser.parse_args();receipts=[]
    groups=[TASK/group/'assessment.json' for group in ('regression','ipc-wakeup-regression')]
    groups.append(TASK/'profile-probes.noindex/pose-ane/assessment.json')
    for p in groups:
        if not p.exists():continue
        summary=json.loads(p.read_text())
        reports=summary if isinstance(summary,list) else summary['runs']
        for report in reports:
            if not report.get('strict_stable_shadow_gate'):continue
            if args.apply:receipts.append(prune(report))
            else:print(report['session'])
    if args.apply:(TASK/'retention.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
