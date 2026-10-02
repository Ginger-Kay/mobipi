"""Bounded R2 checkpoint observer; no GPU work, dispatch, replay, or training.

Consumes durable collection checkpoints. Only the explicitly named Research
files may be changed, and only while the Research tree is clean and its exact
personal origin/main is synchronized. An unknown edit/divergence holds Git
publication without changing collection processes or artifacts.
"""
import argparse,datetime,fcntl,json,os,subprocess,sys,time
from pathlib import Path

STEM='2026-10-02-obc-wam-dr-v04-r2-batched-train-validation'
FILES=[f'08-experiments/reports/{STEM}.md',f'08-experiments/handoff/{STEM}.md','08-experiments/code-registry.md','docs/obc-wam-active-recovery.md','README.md','03-ideas/testable/stage-aware-risk-constrained-option-selection.md']

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True,stderr=subprocess.STDOUT)

def current_document(old,block,state,owner,now):
    marker='<!-- R2_CHECKPOINT_HISTORY -->'
    if marker in old:
        previous,history=old.split(marker,1)
        history=history+'\n\n'+previous
    else:history='\n\n'+old
    head=f"# DR-v0.4-R2 最新执行检查点\n\nupdated_at: `{now}`\nstatus: `{state}`\ncontrol_write_owner: `{owner}`\nreview_status: `pending`\nformal_train_ready: `false`\n"
    return head+block+'\n\n## 历史快照（以下状态仅属于各自时间戳）\n'+marker+history


def publish(run,output,event,final):
    project=run.parents[4];repo=project/'control'
    if git(repo,'remote','get-url','origin').strip()!='git@github-jhk:Ginger-Kay/MM-WAM-Research.git':raise ValueError('unexpected Research origin')
    if git(repo,'branch','--show-current').strip()!='main':raise ValueError('unexpected Research branch')
    if git(repo,'status','--porcelain').strip():raise ValueError('Research dirty; preserve unknown work')
    git(repo,'fetch','--prune','origin')
    if git(repo,'rev-list','--left-right','--count','HEAD...origin/main').split()!=['0','0']:raise ValueError('Research parity changed; no overwrite/sync')
    rows=load(output/'route-outcomes.json');gate=load(output/'mechanical-gate.json');status=load(run/'status.json')
    new=[r for r in rows if not r['outcome_previously_observed']]
    recorded=[r for r in new if r['attempt'] is not None]
    audited=[r for r in new if 'audit' in r];eligible=[r for r in audited if r['machine_eligible_for_gate']]
    groups={r['group_id'] for r in new};complete=sum(all(any(r['group_id']==g and r['route']==route and r['attempt'] for r in new) for route in ('E','D','A')) for g in groups)
    replay_files=list((run/'batches').glob('*/*/route-*-replay.json'));replay_ok=sum(load(p)['reproducible'] is True for p in replay_files)
    review='pending';owner='Research Control' if final else 'Compute Execution'
    now=stamp();counts=dict(new_outcomes=len(recorded),new_replays=len(replay_files),new_reproducible_replays=replay_ok,new_complete_groups=complete,new_machine_audited_routes=len(audited),new_machine_eligible_routes=len(eligible),admitted_seed109_groups=1,training_runs=0,cv_fits=0,model_inference=0,test_routes=0)
    manifest=load(run/'manifest.json');manifest.update(updated_at=now,status=status['state'],counters=counts,latest_summary=str(output),mechanical_gate=gate['gate_status'],formal_train_ready=False,current_control_write_owner=owner)
    write(run/'manifest.json',manifest)
    execution_root=manifest.get('runtime_code_root',str(run/'runtime'))
    analysis_root=manifest.get('analysis_code_root',str(run/'analysis-runtime'))
    title=f'{now} R2 {event}'
    summary=f'新增已记录路线 {len(recorded)}/105、真实回放 {len(replay_files)}（一致 {replay_ok}）、完整新配对 {complete}/35、已完成机器审计 {len(audited)}（合格 {len(eligible)}）。seed109例外另计原train1组；主分母36组108路，新组human_review=pending。机械Gate `{gate["gate_status"]}`，训练/CV/模型推理/test均0。'
    next_step='本批执行已退出；保留全部成功/失败/缺失，交Research审阅视频/资格/Gate并决定后续；禁止自行训练或test。' if final else f'采集队列按原顺序继续，GPU worker上限{load(run/"manifest.json").get("worker_count",1)}；不重复outcome、不按成功率换组。'
    block=f'\n\n## {title}\n\n{summary}\n\n状态 `{status["state"]}`；{next_step} control_write_owner={owner}。\n\n完整108行表 `{output}/route-outcomes.csv`；所有已保存原片精确路径 `{output}/videos.md` / `videos.csv`；首批及异常优先清单 `{output}/priority-review-videos.json`；机械Gate与固定排除seed109敏感性 `{output}/mechanical-gate.json`。缺失保留空值，不对不完整train提前选择best-fixed。原始路径与SHA见JSON；抽帧/机器审计不冒充人审。\n'
    block+=f'\n当前执行代码 `{manifest["code_commit"]}` / `{execution_root}`，CPU审计并发 {manifest.get("cpu_audit_workers",3)}，GPU采集worker{manifest.get("worker_count",1)}。候选监督包 `{output}/paired-supervision-candidate.npz`（X108×1045、y108×5，缺失值+mask）；它不是训练放行。\n'
    if gate['main'] is not None:
        block+=f'\n冻结数值条件：主24train best-fixed={gate["main"]["best_fixed"]}，oracle相对best-fixed/geometry成功Source增量={gate["main"]["oracle_gain_vs_best_fixed"]}/{gate["main"]["oracle_gain_vs_geometry"]}；23train敏感性best-fixed={gate["excluding_seed109_sensitivity"]["best_fixed"]}，数值Gate变化={gate["sensitivity_numerical_gate_changed"]}。非数值资格是否闭合={gate.get("nonnumerical_qualification_closed",False)}；不据敏感性挑主表。\n'
    for rel in FILES[:2]:
        p=repo/rel;p.write_text(current_document(p.read_text(),block,status['state'],owner,now))
    for rel,prefix in [('docs/obc-wam-active-recovery.md','../08-experiments/'),('README.md','08-experiments/')]:
        p=repo/rel;p.write_text(f'{title}：{summary} {next_step} [交接]({prefix}handoff/{STEM}.md)。\n\n'+p.read_text())
    p=repo/'08-experiments/code-registry.md';p.write_text(f'{title}：执行code `{manifest["code_commit"]}` / `{execution_root}`，非学习汇总代码 `{gate["analysis_code_commit"]}` / `{analysis_root}`，仅消费已记录train/validation。{summary} [报告](reports/{STEM}.md)。\n\n'+p.read_text())
    p=repo/FILES[-1];p.write_text(p.read_text()+f'\n\n## {title}\n\n{summary} {next_step} [报告](../../08-experiments/reports/{STEM}.md)。\n')
    changed=git(repo,'diff','--name-only').splitlines()
    if set(changed)!=set(FILES):raise ValueError('unexpected publication scope')
    for rel in FILES:
        data=(repo/rel).read_bytes()
        if len(data)>50*1024**2 or b'\0' in data:raise ValueError('non-document/oversize publication')
    git(repo,'add','--',*FILES);git(repo,'diff','--cached','--check')
    # Recheck remote immediately before creating and publishing this checkpoint.
    git(repo,'fetch','--prune','origin')
    if git(repo,'rev-list','--left-right','--count','HEAD...origin/main').split()!=['0','0']:raise ValueError('remote advanced during write; retain local diff')
    git(repo,'commit','-m',f'Record R2 {event} collection and audit checkpoint')
    git(repo,'push','origin','HEAD:main');git(repo,'fetch','--prune','origin')
    if git(repo,'rev-list','--left-right','--count','HEAD...origin/main').split()!=['0','0'] or git(repo,'status','--porcelain').strip():raise ValueError('post-push delivery not clean/parity')
    return dict(at=now,commit=git(repo,'rev-parse','HEAD').strip(),counts=counts,output=str(output),event=event,final=final)

def snapshot(run,event,deliver,final):
    now=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=run/'gate'/('snapshot-'+now);code=Path(__file__).resolve().parent.parent
    env=dict(os.environ,PYTHONPATH=str(code/'src'),CUDA_VISIBLE_DEVICES='')
    subprocess.run([sys.executable,str(code/'scripts/dr_v04_r2_summarize.py'),'--run',str(run),'--output',str(out)],check=True,env=env)
    if deliver:
        result=publish(run,out,event,final)
        write(run/'gate'/('delivery-'+now+'.json'),result)
    return str(out)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--deliver',action='store_true');p.add_argument('--once',action='store_true')
    p.add_argument('--exit-file',type=Path);p.add_argument('--stage',choices=('pilot','remainder'),default='remainder');a=p.parse_args();run=a.run.resolve()
    lock=(run/'checkpoint-observer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.once:
        print(snapshot(run,'manual-snapshot',a.deliver,False),flush=True);return
    # Start only after the selected dispatcher exists. A stale pilot status must
    # never be mistaken for completion of a not-yet-started remainder queue.
    deadline=time.monotonic()+24*3600;seen=set();publication_held=False
    while time.monotonic()<deadline:
        if not (run/'status.json').exists():time.sleep(15);continue
        status=load(run/'status.json')
        if status.get('stage')!=a.stage:time.sleep(15);continue
        checkpoints=sorted((run/'batches').glob('checkpoint-*.json'))
        events=[p.stem for p in checkpoints if p.stem not in seen]
        if (a.exit_file if a.exit_file else run/(a.stage+'.exit')).exists() and 'dispatcher-exited' not in seen:events.append('dispatcher-exited')
        for event in events:
            final=event=='dispatcher-exited'
            try:
                output=snapshot(run,event,a.deliver and not publication_held,final)
                print(stamp(),event,output,flush=True)
            except BaseException as exc:
                publication_held=True
                write(run/'gate'/('publication-hold-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),dict(event=event,error=repr(exc),at=stamp(),scientific_worker_untouched=True))
                print(stamp(),'publication_hold',repr(exc),flush=True)
            seen.add(event)
            if final:return
        time.sleep(15)
    raise TimeoutError('bounded 24h observer expired; no worker intervention')

if __name__=='__main__':main()
