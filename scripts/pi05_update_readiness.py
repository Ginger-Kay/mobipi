"""Release each task only from the three declared same-version E outcomes."""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run;tasks={}
    for task,slots in [('CloseDrawer',[1,2,3]),('CloseSingleDoor',[4,5,6])]:
        rows=[]
        for slot in slots:
            receipt=next((r/'episodes/relative-fit2-step-2000-adapter-v6').glob(f'slot-{slot:02d}*/engineering-attempt-0/completed.json'),None)
            if receipt is None:rows.append(dict(slot=slot,status='uncompleted',safe_success=False));continue
            q=json.loads(receipt.read_text());assert q['task']==task and q['adapter_version']=='v6' and q['checkpoint_step']==2000
            path=Path(q['attempt'])/'sprint-safety-audit.json';audit=json.loads(path.read_text()) if path.exists() else {}
            rows.append(dict(slot=slot,receipt=str(receipt),status=q['status'],native_success=q['native_success'],audited=path.exists(),safe_success=audit.get('safety_qualified_success',False)))
        count=sum(x['safe_success'] for x in rows);tasks[task]=dict(passed=count>=2 and all(x['status']!='uncompleted' for x in rows),safe_success=count,denominator=3,rows=rows)
    q=dict(at=datetime.now(timezone.utc).isoformat(),controller='frozen_pi05',policy_checkpoint='/share/personal/chensiyu/haokaijiang/MobiWAM/checkpoints/obc-pi05-v1/20261006T181000Z-query-relative-fit2/2000',
        adapter='v6',physical_control_commit='cf2dde5',collection_commit='7249b4e',compatibility='policy/collection-compatibility.json',tasks=tasks,
        purpose='simulation engineering readiness only',formal_train_ready=False,human_review='pending',D_A_release='separate actual continuation qualification required')
    (r/'policy/task-readiness.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(tasks))
if __name__=='__main__':main()
