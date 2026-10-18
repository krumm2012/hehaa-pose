"""Compare SAME-SPLIT box/pose validation metrics; this is not independent accuracy."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    from ultralytics import YOLO
    from racket_pose_detector import model_digest
    parser=argparse.ArgumentParser();parser.add_argument('--original',required=True);parser.add_argument('--candidate',required=True);parser.add_argument('--dataset',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    report={'accuracy_validated':False,'dataset_yaml_sha256':model_digest(args.dataset)}
    for name,path in [('original',args.original),('candidate',args.candidate)]:
        a=YOLO(path).val(data=args.dataset,imgsz=960,device='mps',half=False,batch=4,workers=0,plots=False,save=False,project=str(out.parent),name='quality-'+name,verbose=False)
        report[name]={'pose_map':float(a.pose.map),'pose_map50':float(a.pose.map50),'box_map':float(a.box.map),'box_map50':float(a.box.map50),'checkpoint_sha256':model_digest(path)}
    out.write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
