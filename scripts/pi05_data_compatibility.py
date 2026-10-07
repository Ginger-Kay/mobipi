"""Per-item old-policy compatibility ledger; no old outcome relabeling."""
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();r=a.run;freeze=read(r/'policy/main-component-freeze.json');rows=[]
 for row in [json.loads(x) for x in (r/'inventory/inventory.jsonl').read_text().splitlines() if x]:
  if row['view']!='PI05_historical':continue
  p=Path(row['raw_receipt']);q=read(p);process=read(p.parent/'process.json') if (p.parent/'process.json').exists() else {};policy=read(p.parent/'policy-binding.json') if (p.parent/'policy-binding.json').exists() else {};oldhash=process.get('source_module_sha256',{})
  differences={name:dict(old=digest,new=freeze['behavioral_components'].get(name)) for name,digest in oldhash.items() if name in freeze['behavioral_components'] and digest!=freeze['behavioral_components'][name]}
  old_cp=policy.get('checkpoint');same=old_cp is not None and Path(old_cp).resolve()==Path(freeze['policy_checkpoint']).resolve();horizon=q.get('declared_sim_horizon_seconds',120.);budget=q['status'].startswith('policy_budget_stop_')
  rows.append(dict(record_id=row['record_id'],raw_receipt=str(p),historical_use=row['historical_use'],parent_group=q['parent_group'],historical_config_id=q['config_id'],route=q['route'],native_success_historical=q['native_success'],status_historical=q['status'],policy_checkpoint=old_cp,current_checkpoint=freeze['policy_checkpoint'],policy_equal=same,adapter_equal=q.get('adapter_version')=='v6',old_horizon_seconds=horizon,current_horizon_seconds=300.,old_budget_stop_not300_terminal=bool(budget and horizon<300),changed_file_bindings=differences,exact_common_dependency_equivalence_certified=False,current_posed_start_config_substitution=False,
   old_current_state_distribution='old actual config retained; not renamed to new start01-06',norm_old=str(Path(old_cp)/'custom-norm-stats.json') if old_cp else None,input_history_certificate=policy.get('action_representation'),auxiliary_label_admission=False,head_masks=[False]*5,
   disposition='retained historical/diagnostic; common controller changed and new posed configs differ; no same-component equivalence certificate',early_terminal_fields_preserved_in_original=True,sealed_test_access=False))
 assert len(rows)==100
 (r/'inventory/component-compatibility.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));(r/'inventory/component-compatibility-summary.json').write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),checked=100,auxiliary_admitted=0,originals_unchanged=True,early_terminations_not_declared300_failures=True,source_count_not_inflated=True),indent=2)+'\n');print('COMPATIBILITY',len(rows),'auxiliary admitted0',flush=True)
if __name__=='__main__':main()
