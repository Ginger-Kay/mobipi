"""Read saved inference/action/native receipts and costs; never execute a policy."""
import argparse, csv, hashlib, json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import h5py
import numpy as np

def read(p): return json.loads(Path(p).read_text())
def csvwrite(p, rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();r=a.run
    out=r/'paper-evidence';at=datetime.now(timezone.utc).isoformat()
    frozen=read(r/'policy/main-component-freeze.json');final=read(r/'policy/final-evaluation-freeze.json')
    integrity=[]
    for kind,entries in [('behavior',frozen['behavioral_components']),('native_controller',frozen['native_controller_components']),('task_checker',frozen['task_checker_components']),('source_manifest',frozen['frozen_source_manifests'])]:
        for name,digest in entries.items():
            p=r/'runtime/mobipi'/name if kind=='behavior' else r.parents[3]/'env'/name if kind=='native_controller' else Path(name)
            integrity.append(dict(kind=kind,path=str(p),passed=hashlib.sha256(p.read_bytes()).hexdigest()==digest))
    for kind,binding in final['methods'].items():
        p=Path(binding['checkpoint']);integrity.append(dict(kind='final_'+kind,path=str(p),passed=hashlib.sha256(p.read_bytes()).hexdigest()==binding['sha256']))
    rows=[];chains=[]
    for p in sorted((r/'episodes').glob('*/slot-*/engineering-attempt-0/completed.json')):
        q=read(p);tag=p.parents[2].name;attempt=Path(q['attempt']);steps=q['steps'];queries=q.get('policy_queries',0)
        audit=read(attempt/'sprint-safety-audit.json') if (attempt/'sprint-safety-audit.json').exists() else {}
        semantics=read(p.parent/'route-semantics.json') if (p.parent/'route-semantics.json').exists() else {}
        stop=read(attempt/'qp-protective-stop.json') if (attempt/'qp-protective-stop.json').exists() else {}
        systems=sorted(attempt.glob('qp-system-failure-*.json'));solver=read(systems[0]).get('solver',{}) if systems else {}
        rows.append(dict(tag=tag,slot=q['slot'],task=q['task'],route=q['route'],parent_group=q['parent_group'],config_id=q['config_id'],
            status=q['status'],wall_seconds=(datetime.fromisoformat(q['ended_at'])-datetime.fromisoformat(q['started_at'])).total_seconds(),
            inference_wall_seconds=q.get('query_seconds',0),policy_queries=queries,physics_complete_steps=steps,terminal_sim_seconds=q.get('terminal_duration_s'),
            censored=q.get('censored'),native_success=q['native_success'],actual_base_path_m=q.get('actual_base_path_m'),
            initial_opening=q.get('initial_native_opening'),terminal_opening=q.get('terminal_native_opening'),
            opening_improvement=q['initial_native_opening']-q['terminal_native_opening'] if 'initial_native_opening' in q else None,
            actual_native_collision=audit.get('native_collision'),safety_pass=audit.get('all_safety_pass'),audit_cpu_wall_seconds=audit.get('elapsed_seconds'),
            QP_stop_reason=stop.get('reason'),QP_optimizer_status=solver.get('optimizer_status'),QP_residual=solver.get('residual'),
            D_fresh_query_after_settle=semantics.get('D_fresh_query_after_settle') if q['route']=='D' else None,
            A_native_contact_overlap_seconds=semantics.get('A_native_contact_overlap_maximum_seconds') if q['route']=='A' else None,
            A_contact_translation_m=semantics.get('A_contact_base_translation_m') if q['route']=='A' else None,
            A_historical_strict_gate=semantics.get('historical_strict_A_semantics_observed') if q['route']=='A' else None,
            historical_overlap_is_not_current_qualification=True if q['route']=='A' else None,receipt=str(p)))
        if tag.startswith('dev-') or q['status'].startswith('X_'): continue
        errors=[];sample=None;command_error=None;query_verified=False;query_count=None
        if queries:
            query_count=len(list(p.parent.glob('query-[0-9][0-9][0-9][0-9].npz')))
            if query_count!=queries:errors.append('saved query count mismatch')
            for n in sorted({1,queries}):
                with np.load(p.parent/f'query-{n:04d}.npz',allow_pickle=False) as z:
                    assert z['actions'].shape==z['normalized'].shape==(10,32)
                    assert np.isfinite(z['actions']).all() and np.isfinite(z['normalized']).all()
                    assert z['state'].shape==(32,) and all(z[k].shape==(224,224,3) for k in ['base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb'])
            query_verified=True
        feedback=p.parent/'query-action-feedback.jsonl'
        if steps:
            with h5py.File(attempt/'demo.hdf5','r') as h:
                actions=h['data/demo_0/actions'];assert actions.shape==(steps,12)
                with feedback.open() as fd:
                    for line in fd:
                        x=json.loads(line)
                        if x['policy_nominal'] is not None:
                            sample=x;break
                if sample:
                    command_error=float(np.max(np.abs(actions[sample['step']]-np.asarray(sample['actual_action']))))
                    if command_error>1e-6:errors.append('applied action differs from saved command')
                    with np.load(p.parent/f'query-{sample["query"]:04d}.npz',allow_pickle=False) as z:
                        assert np.allclose(z['actions'][sample['chunk_offset']],sample['policy_nominal'],rtol=0,atol=1e-6)
                        assert np.allclose(z['normalized'][sample['chunk_offset']],sample['policy_raw_normalized'],rtol=0,atol=1e-6)
                        assert float(z['query_sim_time'])<=float(sample['sim_time'])
            if queries and sample is None:errors.append('queried model but no completed manipulation action')
        if q['route']=='D' and queries and semantics.get('D_fresh_query_after_settle') is not True:errors.append('D fresh query receipt absent')
        binding=read(p.parent/'policy-binding.json')
        if Path(binding['checkpoint']).resolve()!=Path(frozen['policy_checkpoint']).resolve():errors.append('policy checkpoint mismatch')
        if q.get('reference_actions_used') or q.get('human_intervention') or q.get('state_injection_during_episode'):errors.append('unexpected supervision or state injection')
        chains.append(dict(tag=tag,slot=q['slot'],config_id=q['config_id'],route=q['route'],queries=queries,saved_queries=query_count,
            complete_native_control_steps=steps,query_samples_verified=query_verified,saved_nominal_to_applied_native_action_error=command_error,
            forward_action_step_observed=bool(query_verified and sample is not None),passed=not errors,errors=errors,
            scope='Saved first/last model queries and first completed manipulation command bound to native HDF5 action; partial/zero-step stops separately retained. No new forward, action, step, replay or render.',receipt=str(p)))
    csvwrite(out/'episode-execution-costs.csv',rows);csvwrite(out/'runtime-chain.csv',chains)
    summary=dict(at=at,closed_slots=len(rows),scientific_wall_seconds=sum(x['wall_seconds'] for x in rows if not x['status'].startswith('X_')),
        inference_wall_seconds=sum(x['inference_wall_seconds'] for x in rows),audit_cpu_wall_seconds=sum(x['audit_cpu_wall_seconds'] or 0 for x in rows),
        terminal_statuses=dict(Counter(x['status'] for x in rows)),runtime_chains=len(chains),runtime_chain_errors=sum(not x['passed'] for x in chains),
        model_source_integrity=integrity,integrity_passed=all(x['passed'] for x in integrity),
        new_forward_action_envstep_replay_render=0,GPU_core_hours='not measured; wall/kernel query receipts only; keeper utilization and memory excluded',
        historical_overlap_threshold='diagnostic only; never an A label admission or success gate',human_review='pending')
    (out/'execution-evidence-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['at','closed_slots','runtime_chains','runtime_chain_errors','integrity_passed']}),flush=True)
    if summary['runtime_chain_errors'] or not summary['integrity_passed']:raise ValueError('evidence verification errors retained; inspect recorded scope before any further action')

if __name__=='__main__':main()
