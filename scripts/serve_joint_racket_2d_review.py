# -*- coding: utf-8 -*-
"""Local source-frame viewer and immutable human contact-review receipts."""
import copy
import argparse
import hashlib
import json
import math
import threading
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs


def validate_contact_review(review,dataset):
    if not isinstance(review,dict) or review.get('schema')!='tennis.joint-racket-contact-review.v1':
        raise ValueError('Unsupported contact review')
    session=next((s for s in dataset['sessions'] if s['id']==review.get('session_id')),None)
    if session is None or review.get('hashes')!=session['hashes']:
        raise ValueError('Source/session hashes do not match')
    if review.get('annotation_mode')!='model_assisted_review' or review.get('independent_reference') is not False:
        raise ValueError('Model-assisted review must remain explicit')
    if not isinstance(review.get('reviewer'),str) or not review['reviewer'].strip():
        raise ValueError('Reviewer identity is required')
    annotations=review.get('events')
    if not isinstance(annotations,dict) or set(annotations)!={str(e['event_id']) for e in session['events']}:
        raise ValueError('Event identities do not match')
    frames={f['frame_id']:f for f in session['frames']}
    for event_id,item in annotations.items():
        if not isinstance(item,dict):raise ValueError('Invalid event review')
        status=item.get('status')
        if status not in ('unreviewed','interval','confirmed','unknown','not_applicable'):
            raise ValueError('Invalid contact decision')
        endpoints=[item.get('start_frame'),item.get('end_frame')];anchor=item.get('anchor_frame')
        if status in ('interval','confirmed'):
            if any(type(fid) is not int or fid not in frames for fid in endpoints) or endpoints[0]>endpoints[1]:
                raise ValueError('Interval must use ordered original source frames')
            if any(frames[fid]['source_seconds'] is None for fid in endpoints):raise ValueError('Unqualified source PTS')
            reported=item.get('interval_pts')
            if not isinstance(reported,list) or len(reported)!=2 or any(type(t) not in (int,float) or not math.isfinite(t) or abs(t-frames[fid]['source_seconds'])>1e-9 for t,fid in zip(reported,endpoints)):
                raise ValueError('Interval PTS must match source')
            if status=='confirmed':
                if type(anchor) is not int or anchor not in frames or not endpoints[0]<=anchor<=endpoints[1]:
                    raise ValueError('Confirmed anchor must lie in reviewed interval')
                if not isinstance(item.get('note'),str) or not item['note'].strip():raise ValueError('Confirmed contact needs visible evidence notes')
            elif anchor is not None:raise ValueError('Interval-only review must not claim a confirmed anchor')
        elif anchor is not None or endpoints!=[None,None] or item.get('interval_pts') is not None:
            raise ValueError('Unreviewed/unknown/not-applicable contact cannot retain anchors')
        if item.get('view') not in ('front','back','both','unclear'):raise ValueError('Invalid evidence view')
        if not isinstance(item.get('note',''),str):raise ValueError('Invalid evidence note')
    annotated=copy.deepcopy(annotations)
    for item in annotated.values():
        anchor=item.get('anchor_frame')
        item['anchor_source_seconds']=frames[anchor]['source_seconds'] if anchor is not None else None
    return {'schema':'tennis.joint-racket-contact-receipt.v1','session_id':session['id'],
            'hashes':session['hashes'],'reviewer':review['reviewer'].strip(),'events':annotated,
            'annotation_mode':'model_assisted_review','independent_reference':False,
            'model_events_modified':False,'accuracy_validated':False,
            'complete':all(r['status']!='unreviewed' for r in annotations.values())}


def saved_reviews(root, dataset):
    """Latest validated user reviews, keyed by source-bound session."""
    latest = {}
    invalid = 0
    for path in sorted((Path(root)/'human_reviews').glob('*.json'), key=lambda p:p.stat().st_mtime_ns):
        try:
            payload = json.loads(path.read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest() != path.stem:
                raise ValueError('Receipt content hash mismatch')
            receipt = validate_contact_review(payload['review'], dataset)
            latest[receipt['session_id']] = payload['review']
        except (ValueError, KeyError, OSError, TypeError):
            invalid += 1
    return latest, invalid


def review_progress(root, dataset):
    """Count only validated saved human receipts, not browser drafts."""
    latest, invalid = saved_reviews(root, dataset)
    counts = {status: 0 for status in ('unreviewed','interval','confirmed','unknown','not_applicable')}
    sessions = []
    for session in dataset['sessions']:
        receipt = latest.get(session['id'])
        session_counts = {status: 0 for status in counts}
        for event in session['events']:
            status = receipt['events'][str(event['event_id'])]['status'] if receipt else 'unreviewed'
            counts[status] += 1
            session_counts[status] += 1
        sessions.append({'session_id':session['id'], 'counts':session_counts,
                         'complete':session_counts['unreviewed']==0})
    return {'total_events':sum(counts.values()), 'saved_status_counts':counts,
            'reviewed_events':sum(counts.values())-counts['unreviewed'],
            'invalid_receipts_ignored':invalid, 'independent_reference':False, 'sessions':sessions}


def serve(directory,port):
    import cv2
    root=Path(directory).resolve();dataset=json.loads((root/'dataset.json').read_text())
    by_id={s['id']:s for s in dataset['sessions']};lock=threading.Lock()
    for s in dataset['sessions']:
        if hashlib.sha256(Path(s['source']).read_bytes()).hexdigest()!=s['hashes']['source_video']:
            raise ValueError('Original source hash changed: '+s['id'])
    @lru_cache(maxsize=24)
    def frame_bytes(sid,fid):
        session=by_id.get(sid)
        if session is None or fid not in {f['frame_id'] for f in session['frames']}:
            raise ValueError('Unknown source frame')
        with lock:
            cap=cv2.VideoCapture(session['source']);ok_seek=cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,frame=cap.read();cap.release()
        if not ok_seek or not ok:raise ValueError('Source frame could not be decoded')
        ok,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,90])
        if not ok:raise ValueError('JPEG encoding failed')
        return encoded.tobytes()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,data,mime='application/json',status=200):
            self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
        def do_GET(self):
            parsed=urlparse(self.path)
            try:
                if parsed.path=='/api/review-status':
                    self.send(json.dumps(review_progress(root,dataset),ensure_ascii=False).encode());return
                if parsed.path=='/api/review-latest':
                    sid=parse_qs(parsed.query).get('session',[''])[0]
                    if sid not in by_id:raise ValueError('Unknown session')
                    latest,_=saved_reviews(root,dataset)
                    self.send(json.dumps({'review':latest.get(sid)},ensure_ascii=False).encode());return
                if parsed.path=='/api/frame':
                    q=parse_qs(parsed.query);self.send(frame_bytes(q['session'][0],int(q['frame'][0])),'image/jpeg');return
                filenames={'/':'index.html','/index.html':'index.html','/dataset.json':'dataset.json','/gaps_ge_3.csv':'gaps_ge_3.csv','/summary.json':'summary.json'}
                name=filenames.get(parsed.path)
                if name is None:self.send(b'Not found','text/plain',404);return
                mime='text/html; charset=utf-8' if name.endswith('.html') else 'text/csv; charset=utf-8' if name.endswith('.csv') else 'application/json'
                self.send((root/name).read_bytes(),mime)
            except (ValueError,KeyError,OSError) as exc:self.send(json.dumps({'error':str(exc)}).encode(),status=400)
        def do_POST(self):
            if self.path!='/api/review':self.send(b'Not found','text/plain',404);return
            origin=self.headers.get('Origin');expected=f'http://127.0.0.1:{self.server.server_port}'
            if self.headers.get('Host')!=f'127.0.0.1:{self.server.server_port}' or (origin is not None and origin!=expected):self.send(b'Invalid local origin','text/plain',403);return
            try:
                if self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('JSON required')
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=128000:raise ValueError('Invalid review size')
                review=json.loads(self.rfile.read(size));receipt=validate_contact_review(review,dataset)
                payload={'review':review,'receipt':receipt,'received_at':datetime.now(timezone.utc).isoformat()}
                body=json.dumps(payload,ensure_ascii=False,indent=2).encode()
                folder=root/'human_reviews';folder.mkdir(exist_ok=True)
                digest=hashlib.sha256(body).hexdigest();path=folder/(digest+'.json')
                with path.open('xb') as f:f.write(body)
                self.send(json.dumps({'saved':path.name,'receipt':receipt},ensure_ascii=False).encode())
            except (ValueError,KeyError,OSError) as exc:self.send(json.dumps({'error':str(exc)},ensure_ascii=False).encode(),status=400)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'Joint racket/body review: http://127.0.0.1:{server.server_port}/',flush=True)
    server.serve_forever()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--port',type=int,default=8770);a=p.parse_args();serve(a.directory,a.port)
