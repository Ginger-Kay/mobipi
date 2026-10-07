"""Bind completed candidate checkpoint bytes and canonical admission lineage."""
import argparse,csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();r=a.run;root=r.parents[3];dest=r/'policy/checkpoint-content-binding.json'
 if dest.exists():raise ValueError('immutable checkpoint binding already present')
 entries={}
 for name,cp in [('old-fit2',root/'checkpoints/obc-pi05-v1/20261006T181000Z-query-relative-fit2/2000'),('corrective1000',root/'checkpoints/obc-pi05-data-v1'/r.name/'integrated-corrective-fit2/1000'),('corrective2000',root/'checkpoints/obc-pi05-data-v1'/r.name/'integrated-corrective-fit2/2000')]:
  files=[]
  for p in sorted(cp.rglob('*')):
   if not p.is_file():continue
   before=p.stat();digest=sha(p);after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
   files.append(dict(path=str(p.relative_to(cp)),bytes=after.st_size,sha256=digest))
  entries[name]=dict(checkpoint=str(cp),files=files,tree_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),bytes=sum(x['bytes'] for x in files));print('HASHED',name,entries[name]['bytes'],flush=True)
 dest.write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),candidates=entries,closed_readonly_checkpoints=True),indent=2)+'\n')
 admitted=read(r/'inventory/policy-demo-admission.json')['records'];original={x['record_id']:x for x in admitted};canonical=read(r/'inventory/canonical-dataset-binding.json');rows=[]
 for episode in canonical['episodes']:
  quality=original[episode['original_record_id']];assert quality['status']=='admitted'
  rows.append(dict(record_id=episode['record_id'],canonical_original_parent=episode['parent_group'],original_configuration_parent_field=quality['parent_group'],task=episode['task'],route=episode['route'],demonstrator_type=episode['demonstrator_type'],raw_quality_receipt=episode['raw_quality_receipt'],raw_hdf5=episode['raw_hdf5'],frames=episode['frames'],windows=episode['windows'],controller_version=episode['controller_version'],canonical_dataset=str(canonical['dataset']),lineage_receipt=str(r/'inventory/canonical-parent-lineage.json'),template_receipt=str(r/'inventory/parent-template-ancestry.json'),future_state_input=False,old_results_become_pi05_labels=False))
 with (r/'inventory/canonical-policy-demo-admission.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 costs=[]
 for name in ['integrated-fit1','integrated-corrective-fit2']:
  d=r/'policy'/name;recipe=read(d/'freeze.json');result=read(d/'result.json');param=read(d/'trainable-parameters.json')
  costs.append(dict(fit_id=name,recipe_compliant=name=='integrated-corrective-fit2',approved_corrective=name=='integrated-corrective-fit2',started_at=result['started_at'],ended_at=result['ended_at'],wall_seconds=(datetime.fromisoformat(result['ended_at'])-datetime.fromisoformat(result['started_at'])).total_seconds(),steps=result['steps'],batch=recipe['batch'],sampling_groups=recipe['parent_groups'],valid_windows=recipe['windows'],trainable_parameters=param['total'],optimizer_state_dtypes=json.dumps(param['optimizer_state_dtypes']),mean_window_exposures=result['effective_window_exposures'],median_synchronized_step_seconds=result['median_t_step_seconds'],base_checkpoint=recipe['base_checkpoint'],new_optimizer=True,frozen_leaves_unchanged=read(d/'freeze-verification.json')['frozen_leaves_unchanged'],task_capability_asserted=False,raw_result=str(d/'result.json')))
 with (r/'paper-evidence/policy-fit-costs.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(costs[0]));w.writeheader();w.writerows(costs)
 print('CANONICAL ADMISSION',len(rows),len(set(x['canonical_original_parent'] for x in rows)),flush=True)
if __name__=='__main__':main()
