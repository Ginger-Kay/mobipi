"""Generate tables and exportable figures from fixed final predictions only."""
import argparse,json,csv,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mobiwam.sim_sprint_learning import SCALES,HEADS,source_weights
from sim_sprint_inventory import csvwrite,write,read

def logrows(p):
    rows=[]
    if not p.exists():return rows
    for line in p.read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:continue
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();r=a.run
    comparison=Path(read(r/'comparison/current.json')['comparison']) if (r/'comparison/current.json').exists() else r/'comparison'
    out=r/'paper-evidence'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');out.mkdir(exist_ok=False)
    table=read(comparison/'baselines.json');csvwrite(out/'offline-baselines.csv',table)
    preds=np.load(comparison/'final-predictions.npz',allow_pickle=False)
    train=np.load(r/'training/train-only.npz',allow_pickle=False);val=np.load(r/'training/development-validation.npz',allow_pickle=False)
    y=np.concatenate([train['y'],val['y']]);mask=np.concatenate([train['mask'],val['mask']]);metrics=[]
    for split in ('train','development-validation'):
        sp=preds['split']==split
        for name in ('MLP','Linear','B3'):
            for j,head in enumerate(HEADS):
                idx=sp&mask[:,j];groups=preds['group_id'][idx];w=source_weights(np.ones(idx.sum(),dtype=bool),groups);w/=w.sum()
                pred=preds[name][idx,j];target=y[idx,j]
                row=dict(split=split,model=name,head=head,valid_rows=int(idx.sum()),valid_sources=len(set(groups)),
                    MAE=float(np.sum(abs(pred-target)*w)),MSE=float(np.sum((pred-target)**2*w)),units='meters' if j==3 else 'seconds' if j==4 else 'unitless')
                if j<2:
                    if name!='B3' and j==0:
                        raw=preds[name+'_raw'][idx,j];bce=np.logaddexp(0,raw)-target*raw
                    else:
                        prob=np.clip(pred,1e-6,1-1e-6);bce=-target*np.log(prob)-(1-target)*np.log(1-prob)
                    row.update(BCE=float(np.sum(bce*w)),Brier=row['MSE'],accuracy=float(np.sum(((pred>=.5)==target)*w)),positive=int(target.sum()),negative=int(len(target)-target.sum()))
                metrics.append(row)
    csvwrite(out/'head-errors.csv',metrics)
    fig,ax=plt.subplots(figsize=(6.4,4.2))
    for name in ('MLP','Linear'):
        logs=logrows(r/'training'/name/'loss.jsonl')
        for p in sorted((r/'training'/name).glob('resume-*/loss.jsonl')):
            restored=read(p.parent/'resume-binding.json')['restored_step'];logs=[x for x in logs if x['step']<=restored]+logrows(p)
        ax.plot([x['step'] for x in logs],[x['loss'] for x in logs],label=name)
    ax.set(xlabel='Optimizer step',ylabel='Train Source-weighted masked loss',yscale='log',title='Fixed recipe, final step 2000');ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'training-loss.png',dpi=200);fig.savefig(out/'training-loss.svg');plt.close(fig)
    validation=[x for x in table if x['split']=='development-validation'];fig,ax=plt.subplots(figsize=(8,4.2));positions=np.arange(len(validation))
    ax.bar(positions-.18,[x['native_success'] for x in validation],width=.36,label='Native success')
    ax.bar(positions+.18,[x['safety_qualified_success'] for x in validation],width=.36,label='Success and safety receipt')
    ax.set_xticks(positions,[x['method'] for x in validation],rotation=35,ha='right');ax.set(ylim=(0,12),ylabel='Sources / 12',title='Offline paired-outcome lookup; reused development validation')
    ax.legend();fig.tight_layout();fig.savefig(out/'offline-selection.png',dpi=200);fig.savefig(out/'offline-selection.svg');plt.close(fig)
    readiness=read(r/'policy/readiness.json');csvwrite(out/'BC-readiness.csv',readiness['tasks'])
    processes=[]
    for name in ('MLP','Linear'):
        p=r/'training'/name;final=read(p/'completed.json');process=read(p/'process.json')
        attempts=[process]+[read(q) for q in sorted(p.glob('resume-*/process.json'))]
        duration=0
        documented_updates=0
        for i,proc in enumerate(attempts):
            if i<len(attempts)-1:
                logs=logrows(p/'loss.jsonl');end=logs[-1]['at'];documented_updates+=logs[-1]['step']
            else:
                end=final['ended_at'];documented_updates+=final['new_updates_this_attempt']
            duration+=(datetime.fromisoformat(end)-datetime.fromisoformat(proc['started_at'])).total_seconds()
        processes.append(dict(model=name,structural_parameters=33733 if name=='MLP' else 5245,
            trainable_parameters=process['trainable_parameters'],frozen_final_step=2000,documented_training_wall_seconds_lower_bound=duration,
            wall_measurement='failed attempt only through last explicit loss timestamp; successful attempt measured to completion; not kernel-only time',
            documented_optimizer_updates_with_recomputed_work=documented_updates,
            successful_segment_seconds=final['gpu_wall_seconds'],checkpoint=final['checkpoint'],checkpoint_sha256=final['checkpoint_sha256'],
            fit_count=1,seed=17,includes_io_wall_and_healthy_occupancy_contention=True,failed_attempts=max(0,len(attempts)-1)))
    csvwrite(out/'parameters-training-time.csv',processes)
    online=[]
    for p in sorted((r/'episodes').glob('*/*/completed.json')):
        q=read(p);safety=Path(q['attempt'])/'sprint-safety-audit.json';s=read(safety) if safety.exists() else {}
        online.append(dict(group_id=q['group_id'],task=q['task'],method=q['method'],route=q['route'],status='executed',native_success=q['native_success'],
            safety_qualified_success=s.get('safety_qualified_success','pending'),reason=q['result']['reason'],steps=q['result']['steps'],
            attempt=q['attempt'],controller='reference fallback',selector_receipt=str(p.parent/'selection.json')))
    for p in sorted((r/'episodes').glob('*/*/not-executed.json')):
        q=read(p);online.append(dict(group_id=p.parents[1].name,method=p.parent.name,status='not_executed',reason=q['reason'],controller='reference fallback'))
    if online:csvwrite(out/'online-results.csv',online)
    claims=[dict(claim='Frozen BC autonomous two-task E/D/A manipulation',support='unsupported',evidence='policy/readiness.json',restriction='5 failed outcomes plus1 engineering missing; D/A not released; command-locked E has native residual'),
        dict(claim='Matched-input head comparison on inherited reference outcomes',support='qualified',evidence='offline-baselines.csv',restriction='paired-outcome lookup; old development validation reused; not BC distribution or test'),
        dict(claim='Learned OBC improves over fixed/geometry/B3',support='unsupported_pending_review',evidence='offline-baselines.csv and online-results.csv',restriction='requires actual measured paired advantage; old formal Gate closed'),
        dict(claim='Simulator oracle geometry reference proof-of-concept',support='qualified_inherited',evidence='R3/R4 original reports; current reference online separately',restriction='L1 only; no deployment perception or true BC autonomous claim'),
        dict(claim='Long-horizon or independent-environment generalization',support='unsupported',evidence='freeze.json',restriction='two related environment families; no extra stage cases yet'),
        dict(claim='Sealed test entirely unread',support='unsupported',evidence='preflight/deviations.jsonl',restriction='one candidate geometry seed103 accidentally read; no test images/outcomes/training/execution')]
    csvwrite(out/'claim-evidence.csv',claims)
    (out/'method-facts.md').write_text('''# Measured method facts

The current learned selector is a fixed CLIP1024 + geometry21 + E/D/A onehot3 head. MLP1048→32→5 and Linear1048→5 use identical seed17, train groups, head masks, scaler, loss, optimizer and2000-step final rule. B3 uses21 geometry dimensions with alpha1. Continuous heads train on unbounded values using MSE. Single-class collision is a constant and excluded from fitting/ranking.

The training and offline lookup executor is the inherited reference-conditioned native feedback controller. Human recordings and frozen BC capability failures remain distinct views. Source supervision is equal within each head, then valid routes, then Sources.24 train/12 reused development-validation Sources are not a blind test.

New development reference proposals use at most1E+5D+1A and current simulator oracle geometry. The microwave closed-progress repair is a separately disclosed data-preparation difference. Any online results use live cameras, CLIP/head selection and native feedback, with no old action replay or per-step Source injection. They remain L1 reference execution.
''')
    (out/'limitations.md').write_text('''# Required limitations

Both BC tasks failed readiness in the implemented action adapter. E zeroed base commands but did not clamp native qpos; small residual generalized motion is recorded. This is not a universal proof that every correctly integrated BC route fails. No policy D history refresh or A continuation qualified, no new BC primary pairs and no L2/L3 readiness.

One sealed-test geometry metadata object was accidentally opened; its images, actions and outcomes were not opened and it was never trained/executed. The incident invalidates a blanket test-unread statement. Wide BC evidence cameras hide the drawer/base; labeled saved-state diagnostic renderings improve visibility and do not constitute a new episode or independent captured camera.

Human failures are never automatic negative labels. Old successes/reference selection carry historical selection bias. New sixteen configs share two families, and old validation labels are development reuse. All training/guard/failure attempts stay visible. MLP ENOSPC resumed from step500 with full optimizer/scheduler/RNG, including lost-update cost. Training loss is not held-out effectiveness. Collision has no positive training support.

Scientific routes are never retried after a valid stop. Engineering slot4 consumed its one repaired attempt and remains unknown. Preflight rejection is a coverage gap, not a task failure. No parameter search, test evaluation, new large foundation model or manuscript/upload action was performed. Formal counters/Gates remain unchanged and review is pending.
''')
    (out/'anonymous-media-checklist.md').write_text('''# Anonymous artifact checklist for Research

- Keep the public attachment free of usernames, host IDs, repository ownership and absolute project paths; preserve private provenance separately.
- Label reference controller, autonomous BC failure, human demonstration and diagnostic saved-state render distinctly.
- Retain speed, pauses, failures, camera occlusion and initial-state reset notes.
- Verify all media captions against videos/videos.csv, native task manifests and exact content hashes.
- Abstract registration, authorship, manuscript claims and final OpenReview upload require Research review; Compute has not submitted.
''')
    write(r/'paper-evidence/current.json',dict(created_at=datetime.now(timezone.utc).isoformat(),packet=str(out),comparison=str(comparison),review_status='pending',formal_train_ready=False))
    print('evidence packet',out,flush=True)

if __name__=='__main__':main()
