"""Measured costs and publication artifacts, including unsupported claims."""
import argparse,csv,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
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
        fits.append(dict(fit_id=d.name,diagnostic=q['diagnostic'],planned_steps=q['steps'],actual_steps=res.get('steps'),
            batch=q['batch'],parents=q['parent_groups'],windows=q['windows'],started_at=q['created_at'],ended_at=res.get('ended_at'),
            wall_seconds=(datetime.fromisoformat(res['ended_at'])-datetime.fromisoformat(q['created_at'])).total_seconds() if res.get('ended_at') else None,
            median_sync_step_seconds=res.get('median_t_step_seconds'),effective_window_exposure=res.get('effective_window_exposures'),
            status=res.get('status','failed' if (d/'failure.json').exists() else 'running'),result=str(p),checkpoint=res.get('checkpoint')))
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
    (out/'method-facts.md').write_text(f'''# PI05-HARNESS-v1 method facts\n\nUpdated {now}. Actual policy: official pi05_base, JAX/OCDBT, dual supported LoRA plus action/time projections, vision/base Transformers frozen. No generative future video model has been trained here.\n\nThree real policy cameras256->224 PIL bilinear; state22+padded10, nominal intent8+padded24. Arm-mount query frame and query-relative EEF variants have separate datasets/checkpoints/control versions. Horizon10, execute5, flow10 at original20Hz physics. Native commanded goals are reconstructed from original controls, rather than future achieved motion.\n\nPolicy-fit inputs comprise10 original safe reference E episodes/10 parents/12 disjoint static segments/3181 windows. Each microwavedoor parent has a moving step excluded by a predeclared zero-base/torso segmentation rule. No chunks cross gaps. Old reference/human results are demonstration qualifications only, never labels of new pi05 success.\n\nExternal geometry/QP controls mobility and constraints; E uses a constant base target, D stows/navigates/settles then requests fresh observations, A mobility triggers only on actual bilateral pad contact plus articulation. Actual checker, native contact mask,0.5mm conservative continuous sweep and strict15mrad joint margin remain. Controller goals and policy transformations are logged. Native actual outcomes so far: {summary['native_success']}; audited safe successes: {summary['safety_qualified_success']}; pending audits: {summary['pending_safety']}.\n\nOBC recipe is fixed: frozen CLIP context1024+21 geometry+3 route indicators, MLP1048->32->5/Linear1048->5/geometry ridge21 alpha1, seed17/fullbatch FP32/2000 updates. Fits require4 supervised train parents and2 routes of the same frozen pi05 components. They remain unrun until actual data and readiness exist.\n\nThe candidate21 fields describe initial geometry and geometric stow/navigation/reach or base-motion prefixes; they do not certify unknown future closed-loop manipulation. View compatibility is a camera-frustum proxy without occlusion prediction. Simulator fixture geometry is an oracle.\n''')
    (out/'limitations.md').write_text('''# Current limitations\n\nTask readiness requires2/3 safe success under one frozen candidate per task; native checker success alone does not qualify. Development versions and failures stay separate. No old sealed test or its roster is opened. Prospective final sources are known-development and share two fixture/layout families and reference-template correlations; they are not independent environment generalization.\n\nD/A geometric preflight is not actual continuation qualification. A requires observed continuous base/arm/articulation overlap with retained contact and at least two real policy chunks. Native action HDF5 alone omits auxiliary controller targets: full query/action sidecars are part of the control protocol, and full action replay has not been claimed. Two real cameras are retained; human motion review is pending.\n\nLoRA loss, save/restore consistency, infrastructure tests and Git delivery do not establish policy ability or OBC improvement. New paired/OBC/final/online units lacking prerequisites are unrun, never filled with old reference results or artificial zeros. Occupancy utilization is operations state, excluded from scientific resources and evidence.\n''')
    claims=[('official_complete_load','supported','policy/full-load-real-forward/result.json'),
        ('native_action_roundtrip','supported','preflight/native-interface-retry-1/nominal-target-roundtrip.json'),
        ('relative_action_roundtrip','supported','preflight/relative-target-roundtrip.json'),
        ('LoRA_freeze_save_restore','supported','policy/20261006T175500Z-relative-diagnostic50/diagnostic-receipt.json'),
        ('task_safety_readiness','unsupported','paper-evidence/policy-development-outcomes.csv'),
        ('D_A_runtime_qualification','unsupported','phase-state.json'),
        ('OBC_selection_gain','unsupported','phase-state.json'),('formal_or_paper_ready','unsupported','phase-state.json')]
    with (out/'claim-evidence.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['claim','status','evidence']);w.writerows(claims)
    (out/'attachment-manifest.json').write_text(json.dumps(dict(at=now,files=[str(p.relative_to(out)) for p in sorted(out.iterdir()) if p.is_file()],publication_status='development evidence, review pending',paired_final_claims=False),indent=2)+'\n')
    print(json.dumps(dict(at=now,policy_fits=len(fits),scientific_summary=summary)))
if __name__=='__main__':main()
