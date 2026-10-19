"""Compare repeated full snapshot encoding with reuse of already encoded JSONL rows."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.snapshot.read_bytes()
    document=json.loads(source)
    rows=[json.dumps(row,ensure_ascii=False,separators=(',',':')) for row in document['frames']]
    header=json.dumps({k:v for k,v in document.items() if k!='frames'},ensure_ascii=False,separators=(',',':'))
    report={'source':str(args.snapshot.resolve()),'source_sha256':hashlib.sha256(source).hexdigest(),
            'frames':len(rows),'scope':'serialization/assembly only; cached rows already written to JSONL; excludes disk I/O'}
    for name,fn in [('full',lambda:json.dumps(document,ensure_ascii=False,separators=(',',':'))),
                    ('reuse',lambda:header[:-1]+',"frames":['+','.join(rows)+']}')]:
        durations=[]
        for _ in range(25):
            start=time.perf_counter_ns()
            encoded=fn()
            durations.append((time.perf_counter_ns()-start)/1e6)
        assert json.loads(encoded)==document
        report[name]={'p50_ms':statistics.median(durations),'max_ms':max(durations),
                      'samples_ms':durations,'output_sha256':hashlib.sha256(encoded.encode()).hexdigest()}
    assert report['full']['output_sha256']==report['reuse']['output_sha256']
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('full','reuse')},indent=2))


if __name__=='__main__':
    main()
