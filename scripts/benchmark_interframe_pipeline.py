"""Sequential, rotated same-source comparisons; never parallelize model runs."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_racket_pose_pipeline import run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--videos', default='/Users/krum5539/Desktop/Camera/2026-09-30')
    parser.add_argument('--clips', nargs='+', default=['48.43.mp4','49.23.mp4','50.03.mp4'])
    parser.add_argument('--roi-policy', choices=('project','disabled'), default='project')
    parser.add_argument('--roi-stream-id', help='本地视频显式绑定的已核对机位')
    parser.add_argument('--repeats', type=int, default=1)
    options = parser.parse_args()
    if options.repeats < 1:
        parser.error('repeats must be positive')
    if options.roi_policy=='project' and not options.roi_stream_id:
        parser.error('project ROI requires --roi-stream-id; disabled explicitly reproduces the old fixture')
    output = Path(options.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    profiles = [('baseline','barrier','queue'), ('shared','barrier','shared_memory'),
                ('interframe','interframe','shared_memory')]
    reports = []
    for repeat in range(options.repeats):
        for index, name in enumerate(options.clips):
            video = Path(options.videos).resolve() / name
            if not video.is_file():
                raise FileNotFoundError(video)
            rotation = (index + repeat) % len(profiles)
            order = profiles[rotation:] + profiles[:rotation]
            if repeat % 2:
                order = list(reversed(order))
            for profile, schedule, ipc in order:
                args = SimpleNamespace(output=str(output/profile), save_video=True, fault_sidecars=False,
                    production_check=True, pose_compute_units=None, display_mode='async', opencv_threads=None,
                    detector_compute_units=None, model=None, acceptance=None, backend=None, compute_units=None,
                    dual=True, inference_schedule=schedule, racket_ipc=ipc,
                    roi_policy=options.roi_policy,roi_stream_id=options.roi_stream_id)
                (output/profile).mkdir(exist_ok=True)
                report = run_case(args, video, 'exclusive', 'buffered', 1, repeat)
                report.update(profile=profile, inference_schedule=schedule, racket_ipc=ipc)
                reports.append(report)
                (output/'matrix.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2))
                if report['returncode'] != 0:
                    raise RuntimeError('Pipeline failed: '+report['session'])


if __name__ == '__main__':
    main()
