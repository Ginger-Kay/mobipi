"""Frozen drawer v2 data, half-weight hierarchy, old2000 fits and selectors."""
import argparse, json, os, random, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from mobiwam.pi05_data_learning import Head, HEADS, SCALES, scale, support, masked_loss, decode
from pi05_drawer_pipeline import read, write, now, sha


def weights(rows,mask,half=True):
    mask=np.asarray(mask,bool)
    if mask.ndim==1:mask=mask[:,None]
    w=np.zeros(mask.shape,np.float64)
    for h in range(mask.shape[1]):
        parents=sorted({r['parent'] for i,r in enumerate(rows) if mask[i,h]})
        for p in parents:
            configs=sorted({r['config_id'] for i,r in enumerate(rows) if r['parent']==p and mask[i,h]})
            cw={c:(.5 if half and next(r['offset'] for r in rows if r['config_id']==c)=='original' else 1.) for c in configs}
            for c in configs:
                idx=[i for i,r in enumerate(rows) if r['parent']==p and r['config_id']==c and mask[i,h]]
                routes=sorted({rows[i]['route'] for i in idx})
                for route in routes:
                    ids=[i for i in idx if rows[i]['route']==route]
                    w[ids,h]=cw[c]/sum(cw.values())/len(parents)/len(routes)/len(ids)
        assert np.isclose(w[:,h].sum(),1. if parents else 0.)
    return w.astype(np.float32)


def dataset(r):
    source=read(r/'training/source-freeze.json');allrows=source['rows'];rows=[];excluded=[];X=[];Y=[];M=[]
    for row in allrows:
        if row['split']!='train':continue
        p=r/'design/inputs'/row['config_id']/(row['route']+'-X.npy')
        if not p.exists():excluded.append(dict(row,input_status='input_unavailable'));continue
        x=np.load(p);assert x.shape==(1048,) and np.isfinite(x).all()
        q=read(row['receipt']);assert q['config_id']==row['config_id'] and q['route']==row['route']
        y=np.full(5,np.nan,np.float32);m=np.zeros(5,bool);y[0]=row['success'];m[0]=True
        for j,key in [(2,'terminal_native_opening'),(3,'actual_base_path_m'),(4,'terminal_duration_s')]:
            v=q.get(key)
            if v is not None and np.isfinite(v):y[j]=np.clip(1-v,0,1) if j==2 else v;m[j]=True
        rows.append(row);X.append(x);Y.append(y);M.append(m)
    X=np.asarray(X,np.float32);Y=np.asarray(Y,np.float32);M=np.asarray(M,bool)
    assert set(x['parent'] for x in rows)<={114,113,144,131,115,151}
    sw=weights(rows,np.ones(len(rows),bool))[:,0];mean=np.sum(X.astype(float)*sw[:,None],axis=0);var=np.sum((X-mean)**2*sw[:,None],axis=0);std=np.sqrt(np.maximum(var,0));std[std<1e-6]=0
    mean=mean.astype(np.float32);std=std.astype(np.float32);hw=weights(rows,M);heads=support(Y,M);heads[1].update(status='unknown',ranking_supported=False,reason='unsupported_for_this_version')
    out=r/'training/dataset-freeze.npz';assert not out.exists()
    np.savez_compressed(out,X=X,y=Y,mask=M,scaler_weights=sw,head_weights=hw,mean=mean,std=std)
    recipe=dict(at=now(),revision=4,old_recipe_reference=str(Path('/share/personal/chensiyu/haokaijiang/MobiWAM/control/08-experiments/reports/2026-10-08-obc-wam-old-fit2-drawer-obc-multistart-tables/obc-recipe.json')),samples_planned=67,samples_used=len(rows),rows=rows,excluded=excluded,train_success=int(Y[:,0].sum()),train_failure=int((Y[:,0]==0).sum()),independent_parents=len({a['parent'] for a in rows}),heads=heads,head_weights=hw.tolist(),scaler_weights=sw.tolist(),mean=mean.tolist(),std=std.tolist(),config_weight={'original':.5,'new_offset':1.},weights_definition='effectivehead: routes mean withinconfig; config weight normalized withinparent; parents equal; available inputs separately for scaler; no repeat0.5',collision='mask all; not trained or selected; five output interface retained',scaler='actual input rows train only; weighted population variance; std<1e-6 set0',algorithm='old recipe unchanged FP32 CPU Head AdamW/fullbatch2000; final2000 only; ridge21 alpha1 weighted-centered SVD',dataset=str(out),dataset_sha256=sha(out),dev_evaluation_accessed=False)
    write(r/'training/recipe.json',recipe);return recipe


def fit(r,kind):
    cfg=read(r/'training/recipe.json');z=np.load(r/'training/dataset-freeze.npz');xs=scale(z['X'],z['mean'],z['std']);y=z['y'];w=z['head_weights'];heads=cfg['heads'];active=[j for j,h in enumerate(heads) if h['status']=='learnable'];out=r/'training'/kind;out.mkdir(exist_ok=False);started=now();clock=time.monotonic()
    if kind=='ridge':
        fits={}
        for j in active:
            ww=w[:,j].astype(float);idx=ww>0;x=xs[idx,1024:1045].astype(float);target=y[idx,j].astype(float)/SCALES[j];ww=ww[idx];xm=np.average(x,axis=0,weights=ww);ym=np.average(target,weights=ww);u,s,vt=np.linalg.svd((x-xm)*np.sqrt(ww[:,None]),full_matrices=False);coef=vt.T@((s/(s*s+1))*(u.T@((target-ym)*np.sqrt(ww))));fits[str(j)]=dict(coef=coef.tolist(),intercept=float(ym-xm@coef),alpha=1.)
        write(out/'ridge.json',dict(heads=heads,mean=z['mean'].tolist(),std=z['std'].tolist(),fits=fits));cp=out/'ridge.json'
    else:
        random.seed(17);np.random.seed(17);torch.manual_seed(17);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=Head(kind);tx=torch.from_numpy(xs);ty=torch.from_numpy(np.nan_to_num(y/SCALES).astype(np.float32));tw=torch.from_numpy(w)
        opt=torch.optim.AdamW([dict(params=[p for n,p in model.named_parameters() if not n.endswith('bias')],weight_decay=.05),dict(params=[p for n,p in model.named_parameters() if n.endswith('bias')],weight_decay=0.)],lr=3e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=2000,eta_min=1e-5)
        with (out/'loss.jsonl').open('x') as f:
            for step in range(1,2001):
                opt.zero_grad(set_to_none=True);loss,terms=masked_loss(model(tx),ty,tw,active);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step();scheduler.step()
                if step%10==0:f.write(json.dumps(dict(at=now(),step=step,loss=float(loss.detach()),lr=opt.param_groups[0]['lr']))+'\n');f.flush()
                if step in [500,1000,1500,2000]:torch.save(dict(step=step,model=model.state_dict(),scaler=dict(mean=z['mean'],std=z['std']),heads=heads,kind=kind,optimizer=opt.state_dict(),scheduler=scheduler.state_dict()),out/f'step{step:04d}.pt')
        cp=out/'step2000.pt'
    write(out/'completed.json',dict(started_at=started,ended_at=now(),elapsed_seconds=time.monotonic()-clock,fit_once=True,steps=2000 if kind!='ridge' else None,checkpoint=str(cp),sha256=sha(cp),recipe=str(r/'training/recipe.json'),evaluation_outcomes_accessed=False))


def available(r):
    methods={}
    for k in ['MLP','Linear','ridge']:
        cp=r/'training'/k/('ridge.json' if k=='ridge' else 'step2000.pt');done=r/'training'/k/'completed.json';methods[k]=dict(status='available' if cp.exists() and done.exists() else 'unavailable',checkpoint=str(cp) if cp.exists() else None,completion=str(done) if done.exists() else None)
    methods['geometry']=dict(status='available',rule='unchanged scene004.geometry_rule_select; offline only')
    write(r/'training/model-freeze.json',dict(at=now(),methods=methods,primary='MLP',scaler=str(r/'training/dataset-freeze.npz'),heads=read(r/'training/recipe.json')['heads'],frozen_before_first_eval=True));return methods


def predict(r,kind,x):
    frozen=read(r/'training/model-freeze.json')['methods'][kind]
    if frozen['status']!='available':return None
    if kind=='ridge':
        q=read(frozen['checkpoint']);xx=scale(x[None],np.asarray(q['mean']),np.asarray(q['std']))[0];p=np.full(5,np.nan)
        for j,h in enumerate(q['heads']):
            if h['status']=='constant':p[j]=h['constant']
            elif h['status']=='learnable':
                f=q['fits'][str(j)];v=float(xx[1024:1045]@np.asarray(f['coef'])+f['intercept']);p[j]=(np.clip(v,0,1) if j<3 else max(v,0))*SCALES[j]
        return p
    q=torch.load(frozen['checkpoint'],map_location='cpu');m=Head(kind);m.load_state_dict(q['model']);m.eval()
    with torch.inference_mode():raw=m(torch.from_numpy(scale(x[None],q['scaler']['mean'],q['scaler']['std']))).numpy()
    return decode(raw,q['heads'])[0]


def select(per,valid,heads):
    legal=[t for t in 'EDA' if valid.get(t,False)]
    if not legal:return dict(route=None,status='route_unavailable',selector_fallback=False)
    active=[j for j,h in enumerate(heads) if j!=1 and h['status'] in ['constant','learnable'] and h.get('ranking_supported',True)]
    pool=[t for t in legal if t in per and all(np.isfinite(per[t][j]) for j in active)]
    if not pool:return dict(route=legal[0],status='fallback_all_predictions_unavailable',selector_fallback=True)
    for j,window,maximize in [(0,.05,True),(2,.05,True),(3,.02,False),(4,1.,False)]:
        if j not in active:continue
        best=(max if maximize else min)(per[t][j] for t in pool);pool=[t for t in pool if per[t][j]>=best-window] if maximize else [t for t in pool if per[t][j]<=best+window]
    return dict(route=pool[0],status='learned_selection',selector_fallback=False)


def freeze_state(r,s):
    target=r/'design/predictions'/s['config_id']/'freeze.json'
    if target.exists():return read(target)
    f=read(r/'design/inputs'/s['config_id']/'features.json');feature_valid={a['route_family']:a['hard_valid'] for a in f['routes']};valid={t:s['hard_valid_routes'].get(t,False) and feature_valid.get(t,False) for t in 'EDA'};models=read(r/'training/model-freeze.json');values={};choices={}
    for kind in ['MLP','Linear','ridge']:
        if models['methods'][kind]['status']!='available':choices[kind]=dict(route=None,status='model_unavailable',selector_fallback=False);continue
        per={t:predict(r,kind,np.load(r/'design/inputs'/s['config_id']/(t+'-X.npy'))) for t in 'EDA' if valid[t]};values[kind]={t:p.tolist() for t,p in per.items()};choices[kind]=select(per,valid,models['heads'])
    selected=f.get('geometry_selection');choices['geometry']=dict(route=next((a['route_family'] for a in f['routes'] if a['candidate_id']==selected and valid[a['route_family']]),None),status='offline_static_rule')
    result=dict(at=now(),state_order=s['state_order'],config_id=s['config_id'],valid_routes=valid,predictions=values,choices=choices,model_freeze=str(r/'training/model-freeze.json'),before_any_state_route_outcome=True,features=str(r/'design/inputs'/s['config_id']/'features.json'));write(target,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--mode',choices=['dataset','fit','freeze'],required=True);p.add_argument('--model',choices=['MLP','Linear','ridge']);a=p.parse_args()
    if a.mode=='dataset':dataset(a.run)
    elif a.mode=='fit':fit(a.run,a.model)
    else:available(a.run)
