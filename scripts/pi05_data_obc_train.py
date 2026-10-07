"""One fixed final2000 neural fit or alpha1 ridge per predeclared data tier."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from mobiwam.pi05_data_learning import HEADS,SCALES,Head,fit_scaler,scale,support,hierarchy_weights,masked_loss

def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--tier',type=int,choices=[1,2],required=True);ap.add_argument('--model',choices=['MLP','Linear','ridge'],required=True);ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    data=a.run/'training'/f'tier-{a.tier}';binding=json.loads((data/'dataset-binding.json').read_text());z=np.load(data/'train-only.npz',allow_pickle=False)
    X=z['X'];y=z['y'];mask=z['mask'];parents=z['parent_group'];configs=z['config_id'];routes=z['route'];valid=z['hard_valid']
    assert X.shape==(len(y),1048) and y.shape==mask.shape==(len(X),5)
    assert set(z['role'])=={'train'} and not set(parents)&set(binding['reserved_development_evaluation_ancestors'])
    participating=set(parents[mask.any(1)]);scaler_eligible=valid & np.array([p in participating for p in parents])
    mean,std=fit_scaler(X,parents,configs,routes,scaler_eligible);xs=scale(X,mean,std)
    weights=hierarchy_weights(parents,configs,routes,mask);heads=support(y,mask)
    active=[j for j,h in enumerate(heads) if h['status']=='learnable']
    out=data/a.model;out.mkdir(exist_ok=False)
    cfg=dict(created_at=now(),tier=a.tier,model=a.model,architecture='1048->32ReLU->5' if a.model=='MLP' else '1048->5' if a.model=='Linear' else '21 geometry weighted ridge alpha1',
        seed=17,FP32=True,steps=2000 if a.model!='ridge' else None,lr=3e-4,eta_min=1e-5,weight_decay=.05,bias_decay=0.,clip=1.,fullbatch=True,
        scales=SCALES.tolist(),heads=heads,active=active,collision_single_class='not fitted or ranked',source_weighting='per head parent->config->route->original episode; mask renormalized each level',
        scaler='train-only participating parents and hard-valid pre-outcome candidates, same hierarchy',mean=mean.tolist(),std=std.tolist(),train_data_sha256=sha(data/'train-only.npz'),
        fixed_checkpoint='final2000 only',independent_initialization=True,evaluation_outcomes_accessed=False,formal_train_ready=False)
    write(out/'recipe.json',cfg);started=now();clock=time.monotonic()
    if a.model=='ridge':
        fits={}
        for j in active:
            w=weights[:,j].astype(np.float64);idx=w>0;x=xs[idx,1024:1045].astype(float);target=y[idx,j].astype(float)/SCALES[j];ww=w[idx]
            xm=np.average(x,weights=ww,axis=0);ym=np.average(target,weights=ww)
            u,s,vt=np.linalg.svd((x-xm)*np.sqrt(ww[:,None]),full_matrices=False)
            coef=vt.T@((s/(s*s+1))* (u.T@((target-ym)*np.sqrt(ww))))
            fits[str(j)]=dict(coef=coef.tolist(),intercept=float(ym-xm@coef),alpha=1.,head=HEADS[j],loss='same historical closed-form ridge regression; binary outputs clipped, no BCE optimizer for ridge')
        write(out/'ridge.json',dict(recipe=cfg,fits=fits))
        write(out/'completed.json',dict(started_at=started,ended_at=now(),model=a.model,fit_once=True,checkpoint=str(out/'ridge.json'),elapsed_seconds=time.monotonic()-clock));return
    random.seed(17);np.random.seed(17);torch.manual_seed(17);torch.set_num_threads(4)
    if a.device.startswith('cuda'):torch.cuda.manual_seed_all(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    model=Head(a.model).to(a.device);tx=torch.as_tensor(xs,device=a.device);ty=torch.as_tensor(np.nan_to_num(y/SCALES).astype(np.float32),device=a.device);tw=torch.as_tensor(weights,device=a.device)
    opt=torch.optim.AdamW([dict(params=[p for n,p in model.named_parameters() if not n.endswith('bias')],weight_decay=.05),dict(params=[p for n,p in model.named_parameters() if n.endswith('bias')],weight_decay=0.)],lr=3e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=2000,eta_min=1e-5)
    write(out/'process.json',dict(at=now(),pid=os.getpid(),argv=sys.argv,device=a.device,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),parameters=sum(p.numel() for p in model.parameters()),python=sys.executable))
    with (out/'loss.jsonl').open('x') as f:
        for step in range(1,2001):
            opt.zero_grad(set_to_none=True);loss,terms=masked_loss(model(tx),ty,tw,active)
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step();sched.step()
            if step%10==0:
                event=dict(at=now(),step=step,loss=float(loss.detach()),lr=opt.param_groups[0]['lr']);f.write(json.dumps(event)+'\n');f.flush()
                if step%100==0:print(json.dumps(event),flush=True)
            if step in [500,1000,1500,2000]:
                torch.save(dict(step=step,model=model.state_dict(),recipe=cfg,scaler=dict(mean=mean,std=std),heads=heads,kind=a.model,optimizer=opt.state_dict(),scheduler=sched.state_dict()),out/f'step{step:04d}.pt')
    checkpoint=out/'step2000.pt';write(out/'completed.json',dict(started_at=started,ended_at=now(),model=a.model,steps=2000,fit_once=True,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),elapsed_seconds=time.monotonic()-clock,evaluation_outcomes_accessed=False))

if __name__=='__main__':main()
