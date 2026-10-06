"""Exactly one MLP and one matched-input Linear fit, with train-only inputs."""
import argparse, hashlib, json, os, random, socket, subprocess, sys, time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from mobiwam.sim_sprint_learning import *

def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--model',choices=['MLP','Linear'],required=True);a=ap.parse_args();r=a.run
    binding=json.loads((r/'training/dataset-binding.json').read_text())
    if binding['controller']!='reference_fallback' and binding['controller']!='frozen_BC':raise ValueError('executor view not frozen')
    z=np.load(r/'training/train-only.npz',allow_pickle=False);X=z['X'];y=z['y'];mask=z['mask'];groups=z['group_id']
    if len(set(groups))<4 or len(set(z['route'][mask.any(1)]))<2:raise ValueError('insufficient training Sources/routes')
    heads=support(y,mask,groups);active=[j for j,h in enumerate(heads) if h['status']=='learnable']
    participating=set(groups[mask[:,active].any(1)])
    included=np.array([g in participating for g in groups]);mean,std=fit_scaler(X[included],groups[included]);xs=scale(X,mean,std)
    out=r/'training'/a.model;out.mkdir(exist_ok=False)
    code=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip()
    frozen=dict(created_at=now(),model=a.model,architecture='Linear(1048,32)-ReLU-Linear(32,5)' if a.model=='MLP' else 'Linear(1048,5)',
        training_seed=17,steps=2000,lr=3e-4,weight_decay=.05,bias_decay=0.,grad_clip=1.,eta_min=1e-5,full_batch=True,FP32=True,
        continuous_loss='MSE on unbounded raw output',binary_loss='BCEWithLogits',loss_weight='head then valid routes per Source then Sources',
        checkpoint_steps=[500,1000,1500,2000],selection='fixed final2000',collision_single_class='skip fitting/ranking; empirical constant',
        scaler='train-only participating Sources, all three preoutcome routes, Source equal population std; std<1e-6=0',
        heads=heads,active=active,scaler_sources=sorted(participating),mean=mean.tolist(),std=std.tolist(),code_commit=code,
        train_dataset=str(r/'training/train-only.npz'),train_sha256=sha(r/'training/train-only.npz'),validation_labels_accessed=False,
        source_view=binding['controller'],torch_module=torch.__file__,formal_train_ready=False)
    write(out/'config.json',frozen)
    random.seed(17);np.random.seed(17);torch.manual_seed(17);torch.cuda.manual_seed_all(17);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    model=ActiveHead(a.model,active).cuda();tx=torch.from_numpy(xs).cuda();ty=torch.from_numpy((y/SCALES).astype(np.float32)).cuda();tm=torch.from_numpy(mask).cuda()
    _,gid=np.unique(groups,return_inverse=True);tg=torch.from_numpy(gid).cuda()
    params=list(model.named_parameters());optimizer=torch.optim.AdamW([
        dict(params=[p for n,p in params if not n.endswith('bias')],weight_decay=.05),
        dict(params=[p for n,p in params if n.endswith('bias')],weight_decay=0.)],lr=3e-4,betas=(.9,.999),eps=1e-8)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=2000,eta_min=1e-5)
    started=now();t0=time.monotonic()
    write(out/'process.json',dict(started_at=started,pid=os.getpid(),argv=sys.argv,python=sys.executable,host=socket.gethostname(),
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),device=torch.cuda.get_device_name(0),trainable_parameters=sum(p.numel() for p in model.parameters()),
        structural_parameters=33733 if a.model=='MLP' else 5245))
    history=[]
    with (out/'loss.jsonl').open('x') as log:
        for step in range(2001):
            optimizer.zero_grad(set_to_none=True);loss,per_head=masked_loss(model(tx),ty,tm,tg,active)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            if step:
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient')
                optimizer.step();scheduler.step()
            if step%10==0:
                with torch.no_grad():post,h=masked_loss(model(tx),ty,tm,tg,active)
                row=dict(at=now(),step=step,loss=float(post),heads={HEADS[j]:float(v) for j,v in h.items()},lr=optimizer.param_groups[0]['lr'])
                history.append(row);log.write(json.dumps(row)+'\n');log.flush()
                if step%100==0:print(json.dumps(row),flush=True)
            if step in (500,1000,1500,2000):
                torch.save(dict(step=step,architecture=a.model,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    scaler=dict(mean=mean,std=std),head_support=heads,active=active,schema=binding,config=frozen,code_commit=code,
                    rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())),out/f'step{step:04d}.pt')
    write(out/'completed.json',dict(started_at=started,ended_at=now(),steps=2000,gpu_wall_seconds=time.monotonic()-t0,final_loss=history[-1]['loss'],
        checkpoint=str(out/'step2000.pt'),checkpoint_sha256=sha(out/'step2000.pt'),validation_predictions=0,formal_train_ready=False))
    print(json.dumps(json.loads((out/'completed.json').read_text())),flush=True)

if __name__=='__main__':main()
