"""Predeclared development ranking; freeze policy and all behavioral inputs."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();r=a.run
    if (r/'policy/main-component-freeze.json').exists():raise ValueError('main components already frozen')
    assert json.loads((r/'design/repaired-features-freeze.json').read_text())['configs']==90
    assert json.loads((r/'design/repaired-starts-v1/repair-summary.json').read_text())['passed']
    assert json.loads((r/'preflight/base-mapping-actual-validation.json').read_text())['passed']
    assert json.loads((r/'policy/integrated-corrective-fit2/freeze-verification.json').read_text())['frozen_leaves_unchanged']
    root=r.parents[3];roster=json.loads((r/'policy/policy-dev-roster.json').read_text());candidates=[]
    versions=[('old-fit2',root/'checkpoints/obc-pi05-v1/20261006T181000Z-query-relative-fit2/2000','dev-oldfit2'),
        ('corrective1000',root/'checkpoints/obc-pi05-data-v1'/r.name/'integrated-corrective-fit2/1000','dev-corrective1000'),
        ('corrective2000',root/'checkpoints/obc-pi05-data-v1'/r.name/'integrated-corrective-fit2/2000','dev-corrective2000')]
    for order,(name,checkpoint,tag) in enumerate(versions):
        rows=[]
        for slot in roster['slots']:
            p=r/'episodes'/tag/f'slot-{slot["development_slot"]:02d}-{slot["config_id"]}'/'engineering-attempt-0/completed.json'
            if not p.exists():raise ValueError('not all predeclared candidate slots closed: '+str(p))
            q=json.loads(p.read_text());attempt=Path(q['attempt']);af=attempt/'sprint-safety-audit.json';audit=json.loads(af.read_text()) if af.exists() else {}
            # Native failure cannot be a safety-qualified success, regardless of
            # an outstanding geometry audit. Positive native success must wait.
            if q['native_success'] and not audit:raise ValueError('positive candidate needs complete original-protection audit')
            rows.append(dict(slot=slot['development_slot'],receipt=str(p),native_success=q['native_success'],safe_success=bool(q['native_success'] and audit.get('all_safety_pass')),
                status=q['status'],path=q.get('actual_base_path_m'),time=q.get('terminal_duration_s'),audit=str(af) if audit else None))
        candidates.append(dict(name=name,checkpoint=str(checkpoint),tag=tag,order=order,rows=rows,safe=sum(x['safe_success'] for x in rows),native=sum(x['native_success'] for x in rows)))
    bestsafe=max(x['safe'] for x in candidates);pool=[x for x in candidates if x['safe']==bestsafe]
    bestnative=max(x['native'] for x in pool);pool=[x for x in pool if x['native']==bestnative]
    common=[slot['development_slot'] for slot in roster['slots'] if all(next(row for row in candidate['rows'] if row['slot']==slot['development_slot'])['safe_success'] for candidate in pool)]
    for candidate in pool:
        selected=[row for row in candidate['rows'] if row['slot'] in common];candidate['common_success_path']=sum(x['path'] for x in selected);candidate['common_success_time']=sum(x['time'] for x in selected)
    chosen=min(pool,key=lambda x:(x['common_success_path'],x['common_success_time'],x['order']))
    code=r/'runtime/mobipi';files=['scripts/pi05_harness_episode.py','scripts/reference_executor.py','src/mobiwam/pi05_adapter.py','src/mobiwam/pi05_motion.py',
        'src/mobiwam/pi05_route.py','src/mobiwam/pi05_A3.py','src/mobiwam/pi05_natural_A.py','src/mobiwam/planner_min.py',
        'src/mobiwam/reference_collision.py','src/mobiwam/reference_prefix_safety.py','src/mobiwam/reference_formal_substep.py']
    assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain'],text=True).strip()
    freeze=dict(at=datetime.now(timezone.utc).isoformat(),status='frozen_before_any_main_outcome',policy_checkpoint=chosen['checkpoint'],selected_candidate=chosen['name'],candidates=candidates,
        ranking='safe successes -> native successes -> path/time on common safe successes -> old/earlier checkpoint',common_success_slots=common,
        nonconformant_first_fit='integrated-fit1 retained as deviation and separate development, not candidate',corrective_authorization=str(r/'policy/corrective-fit-authorization.json'),
        code_commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),behavioral_components={name:sha(code/name) for name in files},
        openpi_commit=subprocess.check_output(['git','-C',str(r/'runtime/openpi'),'rev-parse','HEAD'],text=True).strip(),
        policy_auxiliary={name:sha(Path(chosen['checkpoint'])/name) for name in ['custom-norm-stats.json','policy-adapter-config.json']},
        source_split=str(r/'inventory/source-split.json'),start_design=str(r/'design/start-design.json'),features=str(r/'design/repaired-features-freeze.json'),roster=str(r/'design/primary-roster-plan.json'),
        input_components={'CLIP_revision':'32bd64288804d66eefd0ccbe215aa642df71cc41','input_dimension':1048,'geometry_fields':21,'route_onehot':3,'source_time0_only':True},
        safety={'joint_margin_rad_strictly_greater':.015,'sweep_m':.0005,'QP_tolerance':1e-8,'other_constraints':'unchanged original binding'},
        policy_sampling_seed=20261007,horizon_sim_seconds=300,wall_seconds=2700,base_inverse_actual_proof=str(r/'preflight/base-mapping-actual-validation.json'),formal_train_ready=False)
    (r/'policy/main-component-freeze.json').write_text(json.dumps(freeze,indent=2)+'\n');print(json.dumps({k:freeze[k] for k in ['at','selected_candidate','policy_checkpoint','common_success_slots']}),flush=True)

if __name__=='__main__':main()
