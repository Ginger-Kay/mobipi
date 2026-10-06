"""Measured costs and publication artifacts, including unsupported claims."""
import argparse,csv,json,os
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/share/personal/chensiyu/haokaijiang/MobiWAM/cache/obc-pi05-v1/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();r=a.run;out=r/'paper-evidence';out.mkdir(exist_ok=True)
    now=datetime.now(timezone.utc).isoformat();fits=[]
    for d in sorted((r/'policy').glob('*')):
        f=d/'freeze.json';p=d/'result.json'
        if not f.exists():continue
        q=json.loads(f.read_text());res=json.loads(p.read_text()) if p.exists() else {}
        failure_path=d/'failure.json';failure=json.loads(failure_path.read_text()) if failure_path.exists() else {}
        ended=res.get('ended_at') or failure.get('at')
        measured=d/'metrics.jsonl';partial=[json.loads(line) for line in measured.read_text().splitlines()] if measured.exists() else []
        fits.append(dict(fit_id=d.name,diagnostic=q['diagnostic'],planned_steps=q['steps'],actual_steps=res.get('steps',partial[-1]['step'] if partial else 0 if failure else None),
            batch=q['batch'],parents=q['parent_groups'],windows=q['windows'],started_at=q['created_at'],ended_at=ended,
            wall_seconds=(datetime.fromisoformat(ended)-datetime.fromisoformat(q['created_at'])).total_seconds() if ended else None,
            median_sync_step_seconds=res.get('median_t_step_seconds'),effective_window_exposure=res.get('effective_window_exposures'),
            status=res.get('status','failed' if failure else 'running'),result=str(p),checkpoint=res.get('checkpoint'),
            wall_scope='declared fit start through completion/failure, including initialization/JIT/save',failure_receipt=str(failure_path) if failure else None,
            failure_signature=failure['traceback'].splitlines()[-1] if failure else None))
    with (out/'policy-fit-costs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(fits[0]));w.writeheader();w.writerows(fits)
    fig,ax=plt.subplots(figsize=(7,3.2));present=0
    for d in sorted((r/'policy').glob('*')):
        if 'diagnostic' in d.name or not (d/'metrics.jsonl').exists():continue
        data=[json.loads(x) for x in (d/'metrics.jsonl').read_text().splitlines()]
        if not data:continue
        steps=np.array([x['step'] for x in data]);loss=np.array([x['loss'] for x in data]);assert np.isfinite(loss).all()
        med=np.array([np.median(loss[max(0,i-24):i+1]) for i in range(len(loss))])
        ax.plot(steps[::10],med[::10],label=d.name.split('Z-')[-1]);present+=1
    ax.set(xlabel='Optimizer updates',ylabel='Training loss, trailing25 median',yscale='log',title='Policy fit diagnostics; task readiness is measured separately')
    if present:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'policy-training-loss.png',dpi=180);fig.savefig(out/'policy-training-loss.svg');plt.close(fig)
    summary=json.loads((out/'progress-summary.json').read_text())
    readiness=json.loads((r/'policy/task-readiness.json').read_text())
    obc=[]
    for name in ('MLP','Linear'):
        p=r/'training'/name/'completed.json'
        if p.exists():obc.append(dict(model=name,**json.loads(p.read_text())))
    if obc:
        with (out/'OBC-fit-costs.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(obc[0]));w.writeheader();w.writerows(obc)
        fig,ax=plt.subplots(figsize=(6,3))
        for name in ('MLP','Linear'):
            p=r/'training'/name/'loss.jsonl'
            if p.exists():
                data=[json.loads(line) for line in p.read_text().splitlines()]
                ax.plot([x['step'] for x in data],[x['loss'] for x in data],label=name)
        ax.set(xlabel='Optimizer updates',ylabel='Masked Source-equal training loss',yscale='log',title='Fixed OBC fits; final2000 only');ax.legend()
        fig.tight_layout();fig.savefig(out/'OBC-training-loss.png',dpi=180);fig.savefig(out/'OBC-training-loss.svg');plt.close(fig)
    final=json.loads((out/'final-comparison-status.json').read_text()) if (out/'final-comparison-status.json').exists() else None
    binding=json.loads((r/'training/dataset-binding.json').read_text()) if (r/'training/dataset-binding.json').exists() else None
    facts=dict(at=now,task_readiness={k:dict(passed=v['passed'],safe_success=v['safe_success'],denominator=v['denominator']) for k,v in readiness['tasks'].items()},
        OBC_fits_completed=[x['model'] for x in obc],paired_valid=summary['paired_valid'],online_valid=summary['online_valid'],final=final,
        supervised_train_parents=binding['train_parent_groups'] if binding else None,supervised_train_routes=binding['supervised_train_routes'] if binding else [],
        supervision_cutoff='21:20UTC deterministic time/roster; later outcomes retained but never appended/refitted',human_review='pending',formal_train_ready=False)
    (out/'measured-method-status.json').write_text(json.dumps(facts,indent=2)+'\n')
    audits=[]
    for p in (r/'episodes').glob('*/slot-*/engineering-attempt-*/source-*/*/attempt-*/sprint-safety-audit.json'):
        q=json.loads(p.read_text());audits.append(q['elapsed_seconds'])
    review=list(csv.DictReader((out/'review-copies.csv').open())) if (out/'review-copies.csv').exists() else []
    resources=dict(at=now,policy_fit_rows='policy-fit-costs.csv',OBC_fit_rows='OBC-fit-costs.csv' if obc else None,
        completed_unique_episode_wall_seconds=summary['completed_episode_wall_seconds'],completed_control_steps=summary['completed_control_steps'],
        actual_synchronous_policy_query_seconds=summary['query_seconds'],recorded_safety_audit_CPU_wall_seconds=sum(audits),
        labeled_derivative_process_wall_seconds=sum(float(x['elapsed_seconds']) for x in review),policy_failed_fit_wall_seconds=sum(x['wall_seconds'] or 0 for x in fits if x['status']=='failed'),
        interpretation='sums of process/episode wall duration, including failures; parallel sums are not workflow elapsed time or dedicated GPU-hours; occupancy excluded',
        workflow_started_at='2026-10-06T15:04:49.291352+00:00',workflow_elapsed_seconds=(datetime.now(timezone.utc)-datetime.fromisoformat('2026-10-06T15:04:49.291352+00:00')).total_seconds())
    (out/'resource-costs.json').write_text(json.dumps(resources,indent=2)+'\n')
    (out/'method-facts.md').write_text(f"""# PI05-HARNESS-v1 measured method facts

Updated {now}. The manipulation policy is official pi05_base restored through JAX/OCDBT, with supported dual LoRA and action/time interface projections. Vision and original Transformer weights are frozen. No generative future-video world model was trained. Policy fit is shared preprocessing cost, not an OBC-specific advantage.

Recorded policy cameras are agentview_left, eye_in_hand and agentview_right mapped to the three canonical policy slots; the right slot is not a physical second wrist camera. Images256->224 use PIL bilinear. State22+padding10 and EEF intent8+padding24 are declared in the adapter. Robot sensor base pose is the Panda arm mount. Commanded nominal EEF goals come from original controller commands, rather than future achieved motion. The selected policy uses query-relative EEF targets, horizon10, execute5, flow10 and original20Hz physics.

Policy-fit data:10 safe original E episodes/10 parent groups/12 disjoint static segments/3181 windows. Two microwave records each exclude one base-yaw command by the predeclared segmentation rule; no action chunk crosses a gap. These historical demonstration qualifications are not new pi05 outcome labels. The first5000-step fit and relative2000-step fit are separate; selected checkpoint is fit2/2000.

E locks the base. D stows, navigates, settles, reaches and requests a fresh physical observation/query before locked-base manipulation. D prefix/wiring qualification is distinct from D task success. A1 used bilateral pads. Private A2/A3 use existing-mask permitted finger/target contact and articulation; actual semantics require two fresh chunks, five continuous control overlaps, net5mm or10mrad, and0.25s consecutive native contact/base/arm/fixture overlap. A3 adds measured contact-normal preload through the same QP; original pi05 intention and correction are recorded. Native checker success alone does not qualify A. No future reference manipulation teacher is used.

Original physics, gains, velocity caps, forbidden-contact mask and native checker remain. Safety qualification uses the recorded native0.5mm continuous swept clearance and joint margin strictly above15mrad. Native-response v6 maps the existing PD gains to goal horizons and preserves the original Panda coupled closure with a rigid-self clearance buffer. All versions, failures, actual actions, query transformations and auxiliary controller goals are retained.

OBC inputs are frozen CLIP1024 plus21 preoutcome geometric-prefix features plus three route indicators. The fixed models are MLP1048->32->5, Linear1048->5 and ridge21 alpha1. Seed17, FP32/fullbatch AdamW,2000 updates and fixed final2000 are used without score-driven refits. Completed neural fits: {facts['OBC_fits_completed']}. Five outputs remain structurally present; inactive output rows are frozen buffers. Missing labels are NaN with masks. A single-class collision head is an empirical constant skipped in fitting/ranking. Constant physical path/duration predictions remain in meters/seconds. Scaling uses all three preoutcome inputs of participating train parents only.

Measured task readiness and coverage: {json.dumps(facts['task_readiness'])}. The main experiment is partial CloseDrawer E/D; the microwave task and unqualified A remain explicit unrun coverage. Final models, predictions, all component identities and roster are frozen before the first final outcome. Paired lookup and separately executed online results are distinct tables. Native outcomes {summary['native_success']}, audited safe successes {summary['safety_qualified_success']}, pending safety audits {summary['pending_safety']} include development failures/versions and must not be interpreted as final accuracy.

The21 features measure initial geometry and declared stow/navigation/reach/base prefixes; they do not certify future closed-loop manipulation. View compatibility is a frustum proxy without occlusion prediction. Fixture geometry is simulator-oracle information. Actual results and costs are in final-method-summary.csv, common-success-cost-comparisons.csv, measured-method-status.json and resource-costs.json when available.
""")
    (out/'limitations.md').write_text("""# Evidence limitations

Task readiness requires2/3 audited safe E successes under one frozen candidate. CloseDrawer and microwave readiness are separate. D fresh-query wiring is not D success. A native success without the declared continuous overlap is a semantic failure. Missing A is unrun, never zero-filled, and the partial E/D experiment is not a completed three-route experiment. All ten capability rounds are consumed; no eleventh revision is authorized in this run.

All prospective final groups are known-development and share fixture/layout and historical reference-template correlations. The old sealed roster and outcomes were not accessed. Two final CloseDrawer parents within one family provide descriptive engineering evidence, not significance or independent environment generalization. Half of the minimal16-parent two-task roster is outside the released task scope. The deterministic21:20 training cutoff can leave unequal route coverage; late outcomes remain in the evidence but do not enter the frozen fit.

A crossover or oracle benefit is not assumed. FixedE and all other methods share the exact pi05 and public controls. Equal success results do not support OBC improvement. Each method's successful-subset means are descriptive; claims of speed require the separately listed common-success comparison. Missing outcomes give explicit lower/upper counts. The oracle is restricted to executed routes with resolved safety.

Native action HDF5 omits auxiliary controller goals, so full query/action sidecars belong to the protocol. Full action replay is not claimed. Two actual evidence cameras and full native videos are retained; human motion review remains pending. Loss reduction, tests, Git parity and recorded simulation success do not upgrade formal Gate, readiness approval, manuscript or submission status. Occupancy is operations only and is excluded from scientific evidence and resource attribution.
""")
    claims=[('official_complete_load','supported','policy/full-load-real-forward/result.json'),
        ('native_action_roundtrip','supported','preflight/native-interface-retry-1/nominal-target-roundtrip.json'),
        ('relative_action_roundtrip','supported','preflight/relative-target-roundtrip.json'),
        ('LoRA_freeze_save_restore','supported','policy/20261006T175500Z-relative-diagnostic50/diagnostic-receipt.json'),
        ('CloseDrawer_E_safety_readiness','qualified' if readiness['tasks']['CloseDrawer']['passed'] else 'unsupported','policy/task-readiness.json'),
        ('microwave_E_safety_readiness','qualified' if readiness['tasks']['CloseSingleDoor']['passed'] else 'unsupported','policy/task-readiness.json'),
        ('D_prefix_and_fresh_query','qualified','policy/harness-route-release.json'),
        ('CloseDrawer_A_continuous_cooperation','unsupported','paper-evidence/harness-qualification.csv'),
        ('OBC_fixed_fits','supported' if len(obc)==2 else 'unsupported','paper-evidence/OBC-fit-costs.csv' if obc else 'phase-state.json'),
        ('OBC_selection_gain','unsupported','paper-evidence/final-comparison-status.json' if final else 'phase-state.json'),
        ('formal_or_paper_ready','unsupported','phase-state.json')]
    with (out/'claim-evidence.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['claim','status','evidence']);w.writerows(claims)
    (out/'attachment-manifest.json').write_text(json.dumps(dict(at=now,files=[str(p.relative_to(out)) for p in sorted(out.iterdir()) if p.is_file()],publication_status='development evidence, human review pending',paired_final_evidence=final is not None,
        anonymous_attachment_names='relative filenames listed here; internal absolute provenance/media paths remain in internal manifests'),indent=2)+'\n')
    print(json.dumps(dict(at=now,policy_fits=len(fits),OBC_completed=[x['model'] for x in obc],scientific_summary=summary)))
if __name__=='__main__':main()
