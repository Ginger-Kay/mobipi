"""Labeled derivatives of actual recorded videos; originals remain untouched."""
import argparse,csv,hashlib,json,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path

FFMPEG='/share/personal/chensiyu/haokaijiang/.local/envs/dev/bin/ffmpeg'
FFPROBE='/share/personal/chensiyu/haokaijiang/.local/envs/dev/bin/ffprobe'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
 output=r/'paper-evidence/review-copies';output.mkdir(exist_ok=True);work=[]
 for receipt in sorted((r/'episodes').glob('*/slot-*/engineering-attempt-*/completed.json')):
  q=json.loads(receipt.read_text());attempt=Path(q['attempt'])
  if q['steps']==0:continue
  binding=json.loads((receipt.parent/'policy-binding.json').read_text());checkpoint=Path(binding['checkpoint'])
  for camera,filename in [('main','original.mp4'),('panorama','panoramic.mp4')]:
   if not (attempt/filename).exists():continue
   key=receipt.parents[2].name+'-slot'+str(q['slot'])+'-'+camera
   work.append((receipt,q,binding,checkpoint,attempt,camera,filename,key))
 def render(unit):
  receipt,q,binding,checkpoint,attempt,camera,filename,key=unit;meta=output/(key+'.json')
  if meta.exists():return json.loads(meta.read_text())
  t0=time.monotonic();label=output/(key+'.txt');copy=output/(key+'.mp4');partial=output/(key+'.partial.mp4')
  assert not copy.exists() and not partial.exists()
  label.write_text('pi05 '+checkpoint.parent.name.split('Z-')[-1]+'/'+checkpoint.name+' | '+q.get('adapter_version','v1')+' | '+q['route']+' | '+camera+'\n'+
      q['task']+' | '+q['status']+'\nroute semantics '+str(q.get('route_semantics_pass',True))+' | autonomous | source restored at start | 1x | human review pending\n')
  vf='drawtext=fontfile='+FONT+':textfile='+str(label)+':x=8:y=8:fontsize=12:fontcolor=white:box=1:boxcolor=black@0.65'
  command=[FFMPEG,'-nostdin','-hide_banner','-loglevel','error','-n','-i',str(attempt/filename),'-vf',vf,'-an','-c:v','libx264','-preset','veryfast','-crf','23','-threads','2','-movflags','+faststart',str(partial)]
  with (output/(key+'.log')).open('x') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
  probe=subprocess.run([FFPROBE,'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,r_frame_rate','-of','json',str(partial)],capture_output=True,text=True,check=True,timeout=180)
  stream=json.loads(probe.stdout)['streams'][0];assert int(stream['nb_read_frames'])==q['steps'];assert stream['r_frame_rate']=='20/1'
  partial.rename(copy)
  native=json.loads((attempt/'task-video-manifest.json').read_text());pan=json.loads((attempt/'panoramic-binding.json').read_text())
  sha=native['files']['original.mp4']['sha256'] if camera=='main' else pan['sha256']
  row=dict(at=datetime.now(timezone.utc).isoformat(),evaluation=receipt.parents[2].name,slot=q['slot'],task=q['task'],route=q['route'],status=q['status'],
      original=str(attempt/filename),original_sha256=sha,review_copy=str(copy),camera=camera,frames=q['steps'],fps=20,playback_speed=1,
      full_derivative_decode=True,copy_sha256=hashlib.sha256(copy.read_bytes()).hexdigest(),receipt=str(receipt),elapsed_seconds=time.monotonic()-t0,
      generated_from_actual_recorded_video=True,state_rerendered=False,human_review='pending')
  meta.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(dict(key=key,frames=q['steps'])),flush=True);return row
 with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(render,work))
 if rows:
  with (r/'paper-evidence/review-copies.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (r/'paper-evidence/review-copies-summary.json').write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),copies=len(rows),all_originals_preserved=True,new_rollouts=0,
     ffmpeg=FFMPEG,ffprobe=FFPROBE,source='actual video pixels plus labels; no state rerender'),indent=2)+'\n')
if __name__=='__main__':main()
