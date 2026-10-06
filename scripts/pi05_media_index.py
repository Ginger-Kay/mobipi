"""Reuse native primary decode receipts; fully decode the second real camera."""
import argparse,csv,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import cv2

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();r=a.run
    paper=r/'paper-evidence';paper.mkdir(exist_ok=True);journal=paper/'media-decode-journal.jsonl'
    done={}
    if journal.exists():
        for line in journal.read_text().splitlines():
            q=json.loads(line);done[q['attempt']]=q
    for p in sorted((r/'episodes').glob('*/slot-*/engineering-attempt-*/completed.json')):
        q=json.loads(p.read_text());attempt=Path(q['attempt'])
        if str(attempt) in done:continue
        if not (attempt/'original.mp4').exists():continue
        native=json.loads((attempt/'task-video-manifest.json').read_text());pan=json.loads((attempt/'panoramic-binding.json').read_text())
        main_hashes=native['decoded_frames_sha256'];assert len(main_hashes)==q['steps']
        assert native['binding']['native']['task']==q['task'] and native['binding']['attempt']==str(attempt)
        assert pan['frames']==q['steps'] and pan['native_model_geometry_sha256']==native['binding']['native']['native_geometry_sha256']
        assert sha(attempt/'original.mp4')==native['files']['original.mp4']['sha256']
        assert sha(attempt/'panoramic.mp4')==pan['sha256']
        hashes=[];cap=cv2.VideoCapture(str(attempt/'panoramic.mp4'));assert cap.isOpened()
        while True:
            ok,frame=cap.read()
            if not ok:break
            hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
        cap.release();assert len(hashes)==q['steps']
        assert native['binding']['camera']!=pan['camera'] and hashes[0]!=main_hashes[0]
        trace=[json.loads(x) for x in (attempt/'trace.jsonl').read_text().splitlines()]
        for x,t in zip(pan['frame_bindings'],trace):
            b=t['native_frame_binding']
            for key in ('frame_index','native_model_geometry_sha256','actual_qpos_sha256','actual_qvel_sha256','actual_sim_time'):assert x[key]==b[key]
        row=dict(at=datetime.now(timezone.utc).isoformat(),attempt=str(attempt),task=q['task'],route=q['route'],
            evaluation=p.parents[2].name,status=q['status'],frames=q['steps'],main=str(attempt/'original.mp4'),
            panorama=str(attempt/'panoramic.mp4'),main_sha256=native['files']['original.mp4']['sha256'],panorama_sha256=pan['sha256'],
            main_complete_decode='reused native recorder full decoded-frame receipt',panorama_complete_decode=True,
            cameras_distinct=True,all_frame_state_time_bindings_match=True,human_review='pending')
        with journal.open('a') as f:f.write(json.dumps(row)+'\n')
        done[str(attempt)]=row;print(json.dumps({k:row[k] for k in ('task','evaluation','status','frames')}),flush=True)
    rows=list(done.values())
    if rows:
        with (paper/'all-native-videos.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (paper/'media-index-summary.json').write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),episodes=len(rows),
        real_videos=2*len(rows),human_review='pending',new_rollouts=0,state_rerendered=False),indent=2)+'\n')

if __name__=='__main__':main()
