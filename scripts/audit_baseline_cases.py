"""Audit fixed hashed records; do not confuse coverage with accuracy."""
import argparse, hashlib, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baseline_observations import baseline_profiles, normalized_view_trends

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',default='validation/baseline_cases.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();manifest=json.loads(Path(args.manifest).read_text());result=[]
    for case in manifest['cases']:
        for key,path in case['files'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=case['sha256'][key]:
                raise ValueError('fixed input changed: '+path)
        rows=[json.loads(line) for line in Path(case['files']['frames']).read_text().splitlines() if line.strip()]
        events=json.loads(Path(case['files']['events']).read_text())['events']
        results=[];reasons=Counter();usable=0
        for event in events:
            profile=baseline_profiles(rows,event['start_frame'])
            for view,p in profile['views'].items():
                usable+=int(p['valid']);reasons.update(p['reasons'])
            results.append({'event_id':event['event_id'],'baseline':profile,
                'trends':normalized_view_trends(rows,profile,event['start_frame'],event['end_frame'])})
        result.append({'case':case['id'],'candidate_view_count':len(events)*2,
            'baseline_available_count':usable,'baseline_available_rate':usable/max(1,len(events)*2),
            'rejection_reasons':dict(reasons),'measurement_error':None,'incorrect_output_rate':None,
            'accuracy_status':'unmeasured_missing_independent_labels','events':results})
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'manifest':args.manifest,'cases':result},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='events'} for r in result],ensure_ascii=False))
if __name__=='__main__':main()
