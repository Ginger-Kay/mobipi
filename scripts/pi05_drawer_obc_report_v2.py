"""Incremental lightweight T1/T2/T3 and factual Research handoff; no media scan."""
import argparse, csv, json, shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from pi05_drawer_obc_v2 import snapshot
from pi05_drawer_obc_fit_v2 import predict, select, weights
from pi05_drawer_pipeline import read, write, now
from mobiwam.pi05_data_learning import HEADS
STEM='2026-10-09-obc-wam-drawer-obc-paper-sprint'


def csvwrite(r,name,rows,fields):
    p=r/'tables'/name;tmp=p.with_suffix('.tmp')
    with tmp.open('w') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    tmp.replace(p)


def label(row):
    q=read(row['receipt']);a=[row['success'],None,None,None,None]
    for j,k in [(2,'terminal_native_opening'),(3,'actual_base_path_m'),(4,'terminal_duration_s')]:
        v=q.get(k)
        if v is not None and np.isfinite(v):a[j]=float(np.clip(1-v,0,1)) if j==2 else float(v)
    return a


def offline(r):
    if not (r/'training/model-freeze.json').exists():return dict(status='models_not_yet_frozen',metrics=[],lookup=[])
    freeze=read(r/'training/model-freeze.json');heads=freeze['heads'];sources=read(r/'training/source-freeze.json')['rows'];cache={};metrics=[];predrows=[];lookup=[];support=[]
    for split in ['train','dev']:
        for j,h in enumerate(HEADS):
            rr=[a for a in sources if a['split']==split];vals=[label(a)[j] for a in rr];support.append(dict(split=split,head=h,source_samples=len(rr),label_support=sum(v is not None for v in vals),feature_support=sum((r/'design/inputs'/a['config_id']/(a['route']+'-X.npy')).exists() for a in rr),old_original_references=sum(a['reused'] for a in rr),training_status=heads[j]['status'] if split=='train' else 'report_only'))
    for row in sources:
        xp=r/'design/inputs'/row['config_id']/(row['route']+'-X.npy')
        if not xp.exists():continue
        truth=label(row)
        for kind in ['MLP','Linear','ridge']:
            if freeze['methods'][kind]['status']!='available':continue
            key=(row['config_id'],row['route'],kind)
            if key not in cache:cache[key]=predict(r,kind,np.load(xp))
            p=cache[key]
            for j,h in enumerate(HEADS):
                if j==1 or truth[j] is None or p is None or not np.isfinite(p[j]):continue
                predrows.append(dict(split=row['split'],parent=row['parent'],config_id=row['config_id'],offset=row['offset'],route=row['route'],model=kind,head=h,prediction=float(p[j]),observed=truth[j],absolute_error=abs(float(p[j])-truth[j]),squared_error=(float(p[j])-truth[j])**2))
    live=read(r/'slot-ledger.json')
    for row in live['slots']:
        if row['status']!='completed' or not row['method'].startswith('Fixed-'):continue
        pp=r/'design/predictions'/row['config_id']/'freeze.json'
        if not pp.exists():continue
        frozen=read(pp);truth=label(row)
        for kind,per in frozen['predictions'].items():
            pred=per.get(row['route'])
            if pred is None:continue
            for j,h in enumerate(HEADS):
                if j==1 or truth[j] is None or pred[j] is None or not np.isfinite(pred[j]):continue
                value=float(pred[j]);predrows.append(dict(split='evaluation',parent=row['parent'],config_id=row['config_id'],offset=row['offset'],route=row['route'],model=kind,head=h,prediction=value,observed=truth[j],absolute_error=abs(value-truth[j]),squared_error=(value-truth[j])**2))
    for split in ['train','dev','evaluation']:
        for kind in ['MLP','Linear','ridge']:
            for head in ['success','progress','path','terminal_duration']:
                rows=[a for a in predrows if (a['split'],a['model'],a['head'])==(split,kind,head)]
                if not rows:continue
                ww=weights(rows,np.ones(len(rows),bool),half=False)[:,0];metric='Brier' if head=='success' else 'physical_MAE';value=sum(float(w)*a['squared_error' if head=='success' else 'absolute_error'] for w,a in zip(ww,rows));metrics.append(dict(split=split,model=kind,head=head,metric=metric,value=value,n=len(rows),independent_parents=len({a['parent'] for a in rows}),weighting='route/config/parent equal; no train halfweight in reported metrics'))
    groups={}
    for row in sources:groups.setdefault((row['split'],row['parent'],row['config_id'],row['offset']),{})[row['route']]=row
    ledger=read(r/'slot-ledger.json')
    for s in ledger['states']:
        rr={a['route']:dict(a,split='evaluation') for a in ledger['slots'] if a['state_order']==s['state_order'] and a['method'].startswith('Fixed-') and a['status']=='completed'}
        groups[('evaluation',s['environment_seed'],s['config_id'],s['offset_id'])]=rr
    for (split,parent,config,offset),rr in groups.items():
        if set(rr)!={'E','D','A'}:continue
        ff=r/'design/inputs'/config/'features.json'
        if not ff.exists():continue
        f=read(ff);valid={a['route_family']:a['hard_valid'] for a in f['routes']};permodel={};choices={}
        if split=='evaluation':
            pp=r/'design/predictions'/config/'freeze.json'
            if not pp.exists():continue
            frozen=read(pp);choices={k:v.get('route') for k,v in frozen['choices'].items()};valid=frozen['valid_routes']
        else:
            for kind in ['MLP','Linear','ridge']:
                if freeze['methods'][kind]['status']!='available':continue
                per={t:cache[(config,t,kind)] for t in 'EDA' if (config,t,kind) in cache};choices[kind]=select(per,valid,heads)['route']
            choices['geometry']=next((a['route_family'] for a in f['routes'] if a['candidate_id']==f['geometry_selection']),None)
        choices.update({'Fixed-E':'E','Fixed-D':'D','Fixed-A':'A'})
        for method,t in choices.items():
            if t not in rr:continue
            y=label(rr[t]);lookup.append(dict(split=split,parent=parent,config_id=config,offset=offset,method=method,selected_route=t,success=rr[t]['success'],progress=y[2],path_m=y[3],terminal_duration_s=y[4],complete_EDA_support=True,oracle_success=max(a['success'] for a in rr.values()),evidence='offline paired lookup; not fresh online'))
    lookup_summary=[]
    for split in ['train','dev','evaluation']:
        for method in sorted({a['method'] for a in lookup}):
            rr=[a for a in lookup if (a['split'],a['method'])==(split,method)]
            if not rr:continue
            parent_rates=[np.mean([a['success'] for a in rr if a['parent']==p]) for p in sorted({a['parent'] for a in rr})]
            parent_oracle=[np.mean([a['oracle_success'] for a in rr if a['parent']==p]) for p in sorted({a['parent'] for a in rr})]
            lookup_summary.append(dict(split=split,method=method,n=len(rr),success=sum(a['success'] for a in rr),success_per_completed=np.mean([a['success'] for a in rr]),parent_equal_success=np.mean(parent_rates),parent_equal_oracle=np.mean(parent_oracle),independent_parents=len(parent_rates),evidence='complete EDA offline lookup'))
    csvwrite(r,'T2-selector-summary.csv',lookup_summary,['split','method','n','success','success_per_completed','parent_equal_success','parent_equal_oracle','independent_parents','evidence'])
    csvwrite(r,'T2-prediction-metrics.csv',metrics,['split','model','head','metric','value','n','independent_parents','weighting']);csvwrite(r,'T2-prediction-errors.csv',predrows,['split','parent','config_id','offset','route','model','head','prediction','observed','absolute_error','squared_error']);csvwrite(r,'T2-selector-lookup.csv',lookup,['split','parent','config_id','offset','method','selected_route','success','progress','path_m','terminal_duration_s','complete_EDA_support','oracle_success','evidence']);csvwrite(r,'T3-head-support.csv',support,['split','head','source_samples','label_support','feature_support','old_original_references','training_status']);return dict(status='available',metrics=metrics,lookup_rows=len(lookup),head_support=support)


def comparisons(r):
    d=read(r/'slot-ledger.json');rows=d['slots'];out=[];costs=[];common4=[]
    for s in d['states']:
        rr={a['method']:a for a in rows if a['state_order']==s['state_order']}
        if all(a['status']=='completed' for a in rr.values()):common4.append(s['state_order'])
    for comparator in ['Fixed-D','Fixed-E','Fixed-A']:
        pairs=[]
        for s in d['states']:
            rr={a['method']:a for a in rows if a['state_order']==s['state_order']};a,b=rr['OBC-MLP'],rr[comparator]
            if a['status']!='completed' or b['status']!='completed':continue
            pairs.append((s,a,b))
            if a['success']==b['success']==1:
                costs.append(dict(comparator=comparator,state_order=s['state_order'],parent=s['environment_seed'],offset=s['offset_id'],OBC_path_m=a.get('path_m'),baseline_path_m=b.get('path_m'),delta_path_m=a['path_m']-b['path_m'] if a.get('path_m') is not None and b.get('path_m') is not None else None,OBC_terminal_time_s=a.get('terminal_duration_s'),baseline_terminal_time_s=b.get('terminal_duration_s'),delta_terminal_time_s=a['terminal_duration_s']-b['terminal_duration_s'] if a.get('terminal_duration_s') is not None and b.get('terminal_duration_s') is not None else None))
        parentmeans=[np.mean([a['success']-b['success'] for s,a,b in pairs if s['environment_seed']==p]) for p in [142,143] if any(s['environment_seed']==p for s,a,b in pairs)]
        out.append(dict(comparator=comparator,n=len(pairs),independent_parents=len(parentmeans),OBC_win=sum(a['success']==1 and b['success']==0 for s,a,b in pairs),baseline_win=sum(a['success']==0 and b['success']==1 for s,a,b in pairs),both_success=sum(a['success']==b['success']==1 for s,a,b in pairs),both_failure=sum(a['success']==b['success']==0 for s,a,b in pairs),parent_equal_success_delta=float(np.mean(parentmeans)) if parentmeans else None))
    csvwrite(r,'T1-common-support-comparisons.csv',out,list(out[0]));csvwrite(r,'T1-common-success-costs.csv',costs,['comparator','state_order','parent','offset','OBC_path_m','baseline_path_m','delta_path_m','OBC_terminal_time_s','baseline_terminal_time_s','delta_terminal_time_s']);write(r/'tables/common4.json',dict(states=common4,n=len(common4)));return out,costs,common4


def generate(r,final=False):
    value=snapshot(r);off=offline(r);cmp,costs,common4=comparisons(r);manifest=read(r/'run-manifest.json');at=now();counts=value['counts'];model=read(r/'training/model-freeze.json') if (r/'training/model-freeze.json').exists() else {};recipe=read(r/'training/recipe.json') if (r/'training/recipe.json').exists() else {};events=[json.loads(s) for s in (r/'events.jsonl').read_text().splitlines()] if (r/'events.jsonl').exists() else [];starts=[e for e in events if e['event']=='slot_start'];closes=[e for e in events if e['event']=='scientific_execution_closed'];admission=[e for e in events if e['event']=='ordered_slot_admission_closed'];fits={k:read(r/'training'/k/'completed.json') if (r/'training'/k/'completed.json').exists() else dict(status='unavailable') for k in ['MLP','Linear','ridge']};paired=value['OBC_D'];status='completed_bounded_partial_pending_Research_review' if final else 'running_frozen_model_comparison';base=r/'runtime/control';tables=base/'08-experiments/reports'/f'{STEM}-tables';tables.mkdir(exist_ok=True)
    for p in (r/'tables').iterdir():
        if p.suffix in ['.csv','.json']:shutil.copy2(p,tables/p.name)
    write(tables/'training-freeze-summary.json',dict(recipe=str(r/'training/recipe.json'),samples_planned=67,samples_used=recipe.get('samples_used'),train_success=recipe.get('train_success'),train_failure=recipe.get('train_failure'),head_support=recipe.get('heads'),config_weight={'original':.5,'new_offset':1.},scaler_recomputed=True,model_freeze=model,fits=fits,first_eval_start=starts[0]['at'] if starts else None))
    lines=['# PI05-DRAWER-OBC-v2 固定训练与在线比较','',f'created_at: `{manifest["created_at"]}`',f'updated_at: `{at}`',f'status: `{status}`','review_status: `pending`',f'control_write_owner: `{"Research upon normal Git delivery" if final else "Compute"}`',f'run_id: `{r.name}`',f'execution_closed: `{str(final).lower()}`',f'execution_complete_all48: `{str(counts["completed"]+counts["X"]+counts["unavailable"]==48 and counts["unknown"]==0).lower()}`',f'paper_tables_ready: `{str(final and off["status"]=="available").lower()}`','safety_qualified: `false`','formal_train_ready: `unchanged_false`','Gate_status: `historical_unchanged`','',f'按 [contract](../contracts/{STEM}.md) / [prompt](../prompt/{STEM}.md) 修订4执行。北京时间16:40增量导出、16:50科学停止、17:00交付；唯一配置 `{r}/deadline-config.json`。固定old-fit2/2000、v6/20Hz/horizon10/execute5/flow10/120sim（含D前缀），原保护不变；未补采、未训练policy、未读取sealed test、未改论文。','',f'本版 plan48 守恒：`{json.dumps(counts,ensure_ascii=False)}`。所有成功/科学失败只执行一次；unknown、X、unavailable、unrun独立，未完成不填0。','', '## T1 真实在线执行','', '|方法|成功/完成|可靠覆盖 completed/12|unknown|X|unavailable|unrun|running|fallback|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a in value['methods']:lines.append(f'|{a["method"]}|{a["success"]}/{a["completed"]}|{a["completed"]}/12|{a["unknown"]}|{a["X"]}|{a["unavailable"]}|{a["unrun"]}|{a["running"]}|{a["fallback"]}|')
    lines+=['',f'OBC-D同态共同支持 n={paired["n"]}，父组={paired["independent_parents"]}：OBC胜/D败 {paired["OBC_win"]}，OBC败/D胜 {paired["D_win"]}，同成功 {paired["both_success"]}，同失败 {paired["both_failure"]}；父组内平均再等权成功率差 `{paired["parent_equal_success_difference"]}`。缺失父组不赋0，单父组支持明确受限；嵌套状态不作独立样本显著性检验。','',f'完整四方法共同集 n={len(common4)}，state_order={common4}。E/A各自共同支持及共同任务成功成本见 [比较表]({STEM}-tables/T1-common-support-comparisons.csv) / [成本表]({STEM}-tables/T1-common-success-costs.csv)；不同支持集不能直接排榜，成本仅在双方成功且字段可靠时给出，未执行安全审计。','', '## T2 离线预测与选择','',f'离线状态 `{off["status"]}`。MLP/Linear/ridge的Brier与物理MAE按原report权重（route/config/parent等权，未应用train半权重）分别报train/dev/evaluation；dev仅报告，不进入scaler、训练、选模。完整可靠EDA状态才进行固定路线/selector/oracle事后lookup，fresh online另列。见 [预测指标]({STEM}-tables/T2-prediction-metrics.csv) / [lookup]({STEM}-tables/T2-selector-lookup.csv)。Collision本版不训练、不排序、不补造指标。','', '## T3 来源、冻结与覆盖','',f'固定DATA-v2 train49＋旧原态18＝67（52成功/15失败），dev18＋旧原态6＝24（16成功/8失败）。唯一(parent,offset,route)去重91条，train6/dev2/eval2父组未变。实际train输入入集 `{recipe.get("samples_used")}`；缺输入列入excluded而非全0；head支持见 [矩阵]({STEM}-tables/T3-head-support.csv)。原态采前X直接引用，17个新增train起点一次缓存；geometry21沿旧静态构造，slot_index=候选编号/4，simulator-assisted输入披露，零env.step/零policy forward准备。','', '旧原态config权重0.5、新offset1.0；每head按有效route/config/parent独立重归一，scaler按可用train输入同层级单独归一，不重复乘0.5、不加类别平衡。success含0/1并参与BCE，progress=clip(1-native归一化开度,0,1)，path/2m，terminal time/120s（失败不称time-to-success）。所有mean/std/mask/row权重/head状态在首步前新算冻结。MLP1048→32→5、Linear1048→5各独立seed17/CPU FP32旧AdamW固定2000；ridge21 alpha1一次旧weighted-centered SVD；只用2000，不扫参、不早停、不dev选模。','',f'实际fit完成记录：`{json.dumps(fits)}`。模型/所有对照状态冻结 `{model.get("at")}`；首个eval `{starts[0]["at"] if starts else None}`。MLP预指定主模型，Linear/ridge/geometry仅离线；逐态预测在该态任何route outcome前冻结，每态D→OBC fresh restore/inference→E→A。路线全预测坏时确定性hard-valid E<D<A fallback单列，没有legal路线/模型标unavailable。','',f'逐态0/1/null/未执行及双视角原片/HDF5/初态URI见 [T3矩阵]({STEM}-tables/T3-state-method-matrix.csv)，原片留服务器，未全解码/离线安全审计。两eval父组seed142/143早已暴露；12个计划变换均与旧STRESS计划起点重叠，本版独立prospective frozen-model补充，不称新盲测，不拼旧结果。','', '## 停止、provenance与可支持主张','',f'科学收口时间 `{closes[-1]["at"] if closes else None}`；最后准入决定 `{json.dumps(admission[-1] if admission else None)}`。ETA引用同执行链/同两worker DATA-v2完整120sim实际耗时，启动估计895.832s（含60s启动余量），只用新完整120sim记录保守上调，剔除离线审计；save_margin120s，逐slot准入，无固定提前45分钟/整批门，不挑快槽。','',f'Research输入 `{manifest["research_commit"]}`；本版Mobipi `{manifest["code_commit"]}`（branch codex/pi05-drawer-obc-v2-20261009），OpenPI `{manifest["openpi_commit"]}`只读复用，环境/commands/PID见 `{r}/run-manifest.json` 与 launch。隔离worktree保留共享未知dirty/untracked。首次artifact mkdir因共享盘ENOSPC失败（0训练/0eval）；用户清理后同run接续，未删旧证据。','',f'本版OBC-D可靠配对 n={paired["n"]}、父组={paired["independent_parents"]}，父组等权成功差 `{paired["parent_equal_success_difference"]}`；这个数字及四格只支持当前已暴露父组、固定起点和实际覆盖上的描述比较。无配对或无正差时不支持超越Fixed-D；即便有正差，也不支持广泛泛化、显著性、严格A、安全合格、部署、formal/Gate或paper-ready。表格闭合与方法有效分开；人审及论文claim由Research决定。','',f'Mac端ChatGPT请读取本report、同stem handoff及T1/T2/T3表，复核共同支持分母、负/正结果与两个已暴露父组限制，再决定论文使用的主张。最终Git commit/push/parity见 `{r}/delivery/git-delivery.json`；交付成功不代表研究审核。','']
    (base/'08-experiments/reports'/f'{STEM}.md').write_text('\n'.join(lines))
    handoff='\n'.join(['# PI05-DRAWER-OBC-v2 交接入口','',f'created_at: `{manifest["created_at"]}`',f'updated_at: `{at}`',f'status: `{status}`','review_status: `pending`','',f'见完整 [report](../reports/{STEM}.md) 与同stem T1/T2/T3。plan48：`{json.dumps(counts)}`；OBC-D共同支持 `{json.dumps(paired)}`。','',f'服务器 `{r}`。恢复事实入口：run-manifest、phase-state、slot-ledger、continuity-receipt、training/recipe/model-freeze、design/predictions/*/freeze。固定半权重/scaler/模型/12态48槽/顺序与UTC08:50科学停止、09:00交付，不重训、不混旧表、不重跑可靠outcome；机械unknown保留实际修复lineage。','', 'train67/dev24来源与parent split保持；碰撞退出训练/选择；独立fresh OBC。旧STRESS起点重叠/两已暴露父组与simulator-assisted输入限制保留，无安全合格/formal/Gate/方法有效或paper-ready自动升级。','',f'Mac端读取report及表格，复核配对支持和claim。正常Git交付收据 `{r}/delivery/git-delivery.json`，任务专属进程最终核验及GPU operational记录另存delivery/operations，不属于实验evidence。',''])
    (base/'08-experiments/handoff'/f'{STEM}.md').write_text(handoff)
    for rel in ['docs/obc-wam-active-recovery.md','08-experiments/README.md','08-experiments/code-registry.md']:
        p=base/rel;txt=p.read_text();start='<!-- PI05-DRAWER-OBC-v2 live begin -->';end='<!-- PI05-DRAWER-OBC-v2 live end -->'
        if start in txt:txt=txt.split(end,1)[1].lstrip('\n')
        prefix='../08-experiments/' if rel.startswith('docs/') else ''
        msg=f'{at} PI05-DRAWER-OBC-v2 {status}：plan48/completed{counts["completed"]}/unknown{counts["unknown"]}/X{counts["X"]}/unavailable{counts["unavailable"]}/unrun{counts["unrun"]}/running{counts["running"]}；train67/dev24、原态0.5、新态1、一次固定2000；OBC-D paired{paired["n"]}，父组{paired["independent_parents"]}，差{paired["parent_equal_success_difference"]}；UTC08:50/09:00，安全/formal/Gate不升级。Mobipi {manifest["code_commit"]}。 [report]({prefix}reports/{STEM}.md) / [handoff]({prefix}handoff/{STEM}.md)'
        p.write_text(start+'\n'+msg+'\n'+end+'\n\n'+txt)
    manifest.update(updated_at=at,status=status,actual_counts=counts,paper_tables_ready=bool(final and off['status']=='available'),execution_closed=final,science_launched=bool(starts));write(r/'run-manifest.json',manifest);write(r/'delivery/report-export.json',dict(at=at,final=final,counts=counts,offline=off['status'],report=str(base/'08-experiments/reports'/f'{STEM}.md')));print(json.dumps(dict(at=at,final=final,counts=counts,OBC_D=paired)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--final',action='store_true');a=p.parse_args();generate(a.run,a.final)
