"""Close all frozen E denominators; failures cannot authorize BC D/A batches."""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();r=a.run
    roster=json.loads((r/'policy/ability-roster.json').read_text())['slots'];completed=[]
    for slot in roster:
        receipts=[json.loads(p.read_text()) for p in (r/'policy').glob(f"ability-{slot['slot']:02d}-*/completed.json")]
        if len(receipts)!=1:raise ValueError(f"slot {slot['slot']} not closed exactly once: {len(receipts)}")
        x=receipts[0];guard=json.loads((Path(x['attempt'])/'joint-margin-monitor.json').read_text())
        sweep_path=Path(x['attempt'])/'sprint-safety-audit.json'
        sweep=json.loads(sweep_path.read_text()) if sweep_path.exists() else {}
        x['safety_qualified_success']=bool(x['native_success'] and sweep.get('clearance_pass') and guard['minimum']['margin_rad']>.015)
        completed.append(x)
    tasks=[]
    for task in ('CloseDrawer','CloseSingleDoor'):
        rows=[x for x in completed if x['task']==task];native=sum(bool(x['native_success']) for x in rows);safe=sum(x['safety_qualified_success'] for x in rows)
        if native>=2 and any(x['native_success'] and not (Path(x['attempt'])/'sprint-safety-audit.json').exists() for x in rows):
            raise ValueError('native success can meet E condition; actual swept audits required before readiness classification')
        tasks.append(dict(task=task,planned_E=3,completed_E=len(rows),native_success=native,safety_qualified_success=safe,
            E_pass=safe>=2,D_history_readiness='not_run_E_capability_not_passed' if safe<2 else 'requires_primary_slot_check',
            A_chunk_readiness='not_run_E_capability_not_passed' if safe<2 else 'requires_primary_slot_check',BC_batch_released=False))
    if any(x['E_pass'] for x in tasks):raise ValueError('E-passing task requires explicit frozen D/A continuation checks before fallback decision')
    result=dict(created_at=datetime.now(timezone.utc).isoformat(),tasks=tasks,episodes=completed,main_controller='reference_fallback',
        reason='both tasks fail the frozen E capability condition; no BC task qualifies for D/A continuation batch',
        capability_routes_charged=6,primary_budget_remaining=42,new_policy_main_pairs=0,source_init='policy-init-v1 zero-action padded current observations',
        formal_train_ready=False,highest_policy_evidence='real native closed-loop failure evidence; no L2 competence')
    (r/'policy/readiness.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(tasks),flush=True)

if __name__=='__main__':main()
