"""Verify both real cameras and make complete annotated1x viewing copies."""
import argparse,csv,hashlib,html,json,os
from datetime import datetime,timezone
from pathlib import Path
import subprocess,sys,time

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def probe(p):return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames,r_frame_rate,width,height','-of','json',str(p)],text=True))['streams'][0]
def update(r):
 subprocess.run([sys.executable,str(Path(__file__).parent/'pi05_media_index.py'),'--run',str(r)],check=True)
 out=r/'paper-evidence/watch-1x';out.mkdir(exist_ok=True);rows=[]
 for p in sorted((r/'episodes').glob('*/slot-*/engineering-attempt-0/completed.json')):
  q=read(p);attempt=Path(q['attempt'])
  if q['steps']==0:continue
  tag=p.parents[2].name;name=tag+'-slot-'+str(q['slot']);folder=out/name;folder.mkdir(exist_ok=True);binding=folder/'watch-binding.json'
  if binding.exists():rows.extend(read(binding)['views']);continue
  views=[]
  for camera,filename in [('primary','original.mp4'),('panorama','panoramic.mp4')]:
   original=attempt/filename;copy=folder/(camera+'-1x.mp4');caption='PI05-DATA | '+tag+' | '+q['task']+' | '+q['route']+' | '+q['status']+' | autonomous | '+camera+' | 1x20Hz'
   textfile=folder/(camera+'-caption.txt');textfile.write_text(caption+'\n')
   filter='drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:textfile='+str(textfile)+':x=8:y=8:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.7'
   subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(original),'-vf',filter,'-c:v','libx264','-threads','2','-crf','18','-an',str(copy)],check=True)
   source_probe=probe(original);copy_probe=probe(copy);assert source_probe==copy_probe and int(copy_probe['nb_read_frames'])==q['steps'] and copy_probe['r_frame_rate']=='20/1'
   views.append(dict(tag=tag,slot=q['slot'],task=q['task'],route=q['route'],status=q['status'],native_success=q['native_success'],camera=camera,original=str(original),original_sha256=sha(original),viewing_copy=str(copy),viewing_sha256=sha(copy),frames=q['steps'],fps='20/1',playback='1x original native control20Hz',raw_receipt=str(p),human_review='pending',state_rerender=False))
  binding.write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),views=views,complete_native_media=True,new_scientific_execution=False,originals_retained=True),indent=2)+'\n');rows.extend(views);print('MEDIA',name,q['steps'],flush=True)
 if rows:
  with (r/'paper-evidence/all-1x-videos.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 content=['<!doctype html><meta charset="utf-8"><title>PI05 DATA native evidence</title><h1>Complete native video review</h1><p>Autonomous frozen policy;1x20Hz. Original movies and full failure denominator retained. Human review pending.</p>']
 for row in rows:
  content.append('<section><h3>'+html.escape(row['tag']+' / '+row['task']+' '+row['route']+' '+row['camera']+' '+row['status'])+'</h3><video controls preload="none" width="640" src="'+html.escape(os.path.relpath(row['viewing_copy'],r/'paper-evidence'))+'"></video><p><a href="'+html.escape(os.path.relpath(row['original'],r/'paper-evidence'))+'">Original</a> | <a href="'+html.escape(os.path.relpath(row['raw_receipt'],r/'paper-evidence'))+'">Receipt</a></p></section>')
 (r/'paper-evidence/review.html').write_text('\n'.join(content))
 (r/'paper-evidence/watch-summary.json').write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),native_episodes=len(rows)//2,real_views=len(rows),complete_1x_copies=len(rows),human_review='pending',generative_or_state_rerender_media=0),indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--watch',action='store_true');a=ap.parse_args();r=a.run;deadline=datetime.fromisoformat(read(r/'run-manifest.json')['hard_science_deadline'])
 while True:
  update(r)
  if not a.watch or (r/'delivery/stop-media-watch.json').exists() or datetime.now(timezone.utc)>=deadline:break
  time.sleep(30)
if __name__=='__main__':main()
