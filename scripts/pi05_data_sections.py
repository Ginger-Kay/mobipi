"""Extract the declared120s cut from closed actual traces, without replay."""
import argparse,csv,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();rows=[]
 for p in sorted((a.run/'episodes').glob('*/slot-*/engineering-attempt-0/completed.json')):
  q=json.loads(p.read_text());attempt=Path(q['attempt']);f=attempt/'trace.jsonl';row=dict(tag=p.parents[2].name,slot=q['slot'],task=q['task'],route=q['route'],config_id=q['config_id'],parent_group=q['parent_group'],terminal_status=q['status'],terminal_duration_s=q['terminal_duration_s'],requested_section_s=120.,raw_receipt=str(p),raw_trace=str(f),new_forward_or_step=False)
  selected=None;path=0.;origin=None
  if f.exists():
   with f.open() as trace:
    for line in trace:
     x=json.loads(line)
     if origin is None:origin=x['before']['sim_time']
     path+=float(np.linalg.norm(np.asarray(x['after']['base_pos'])[:2]-np.asarray(x['before']['base_pos'])[:2]))
     elapsed=x['after']['sim_time']-origin
     if elapsed>=120.-1e-6:selected=(x,elapsed,path);break
  if selected:
   x,t,path=selected;opening=list(x['after']['target'].values());assert len(opening)==1
   row.update(section_status='actual120s_native_section',actual_elapsed_s=t,control_frame=x['step'],native_success=bool(x['after']['success']),native_opening=opening[0],native_progress=float(np.clip(1-opening[0],0,1)),actual_base_path_prefix_m=path,actual_qpos=x['after']['qpos'],actual_qvel=x['after']['qvel'])
  else:row.update(section_status='terminated_before120s' if q['steps']>0 else 'X_no_action',actual_elapsed_s=None,control_frame=None,native_success=None,native_opening=None,native_progress=None,actual_base_path_prefix_m=None)
  (p.parent/'native-120s-section.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
 out=a.run/'paper-evidence';(out/'native-120s-sections.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
 simplified=[{k:v for k,v in x.items() if k not in ['actual_qpos','actual_qvel']} for x in rows]
 if simplified:
  with (out/'native-120s-sections.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(simplified[0]));w.writeheader();w.writerows(simplified)
 print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),closed_slots=len(rows),actual_sections=sum(x['section_status']=='actual120s_native_section' for x in rows),other_terminations_retained=True)))
if __name__=='__main__':main()
