"""Compare local decoder first reads without changing pipeline clocks or models."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import cv2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-dir',type=Path,default=Path('/Users/krum5539/Desktop/Camera/2026-09-30'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    variants=['configured','threads2','rewind_warmup']
    samples=[]
    references={}
    for repeat in range(args.repeats):
        for clip in sorted(args.video_dir.glob('*.mp4')):
            order=variants[repeat%3:]+variants[:repeat%3]
            for variant in order:
                start=time.perf_counter_ns()
                cap=(cv2.VideoCapture(str(clip),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2])
                     if variant=='threads2' else cv2.VideoCapture(str(clip)))
                open_ms=(time.perf_counter_ns()-start)/1e6
                if not cap.isOpened():
                    raise ValueError('Failed capture: '+str(clip))
                backend=cap.getBackendName()
                threads=cap.get(cv2.CAP_PROP_N_THREADS)
                warmup_ms=0.
                try:
                    if variant=='rewind_warmup':
                        start=time.perf_counter_ns()
                        ok,frame=cap.read()
                        if not ok or not cap.set(cv2.CAP_PROP_POS_FRAMES,0):
                            raise ValueError('Cannot rewind first source frame')
                        warmup_ms=(time.perf_counter_ns()-start)/1e6
                    durations=[];identity=[]
                    for fid in range(25):
                        start=time.perf_counter_ns();ok,frame=cap.read()
                        durations.append((time.perf_counter_ns()-start)/1e6)
                        if not ok:
                            raise ValueError('Incomplete first 25 frames')
                        identity.append({'frame_id':fid,'pts_ms':cap.get(cv2.CAP_PROP_POS_MSEC),
                                         'pixels_sha256':hashlib.sha256(frame.tobytes()).hexdigest()})
                finally:
                    cap.release()
                if clip.name not in references:
                    references[clip.name]=identity
                if identity!=references[clip.name]:
                    raise ValueError('Source pixels/PTS changed: '+clip.name+' '+variant)
                samples.append({'clip':clip.name,'repeat':repeat,'variant':variant,'backend':backend,
                    'reported_threads':threads,'open_ms':open_ms,'warmup_extra_ms':warmup_ms,
                    'first_read_ms':durations[0],'next_24_mean_ms':statistics.mean(durations[1:]),
                    'read_ms':durations,'first25_pixels_and_pts_match':True})
    report={'scope':'Isolated decoding; not whole-pipeline throughput or accuracy acceptance',
            'samples':samples,'identities':references,'summary':{}}
    for variant in variants:
        rows=[r for r in samples if r['variant']==variant]
        report['summary'][variant]={'samples':len(rows),
            'median_first_read_ms':statistics.median(r['first_read_ms'] for r in rows),
            'median_next_24_mean_ms':statistics.median(r['next_24_mean_ms'] for r in rows),
            'median_warmup_extra_ms':statistics.median(r['warmup_extra_ms'] for r in rows)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':
    main()
