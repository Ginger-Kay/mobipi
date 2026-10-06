"""Frozen matched-input ablation, masked Source weighting and selection."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import random
from mobiwam.dr_v04_r3_learning import source_weights, support, fit_scaler, scale

SCALES=np.array([1.,1.,1.,2.,120.])
HEADS=['success','collision','progress','base_path_m','terminal_duration_s']
ROUTES=['E','D','A']

def restore_fit(model,optimizer,scheduler,checkpoint):
    model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
    rng=checkpoint['rng'];torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy']);random.setstate(rng['python'])
    if torch.cuda.is_available() and rng['cuda']:torch.cuda.set_rng_state_all(rng['cuda'])
    return int(checkpoint['step'])

def require_group_split(train, validation):
    if set(train)&set(validation):raise ValueError('Source appears in both splits')

def masked_loss(raw,y,mask,groups,active):
    heads={}
    for j in active:
        idx=mask[:,j]
        if not bool(idx.any()):continue
        values=F.binary_cross_entropy_with_logits(raw[idx,j],y[idx,j],reduction='none') if j<2 else (raw[idx,j]-y[idx,j]).square()
        heads[j]=torch.stack([values[groups[idx]==g].mean() for g in torch.unique(groups[idx])]).mean()
    return (torch.stack(list(heads.values())).mean() if heads else raw.sum()*0),heads

class ActiveHead(nn.Module):
    def __init__(self,architecture,active):
        super().__init__();self.architecture=architecture
        self.hidden=nn.Sequential(nn.Linear(1048,32),nn.ReLU()) if architecture=='MLP' else nn.Identity()
        initial=nn.Linear(32 if architecture=='MLP' else 1048,5)
        self.register_buffer('initial_weight',initial.weight.detach().clone())
        self.register_buffer('initial_bias',initial.bias.detach().clone())
        self.register_buffer('active',torch.tensor(active,dtype=torch.long))
        self.weight=nn.Parameter(initial.weight.detach()[active].clone());self.bias=nn.Parameter(initial.bias.detach()[active].clone())
    def forward(self,x):
        h=self.hidden(x)
        return F.linear(h,self.initial_weight,self.initial_bias).index_copy(1,self.active,F.linear(h,self.weight,self.bias))

def transform(raw,heads):
    pred=np.asarray(raw,dtype=float).copy()
    for j,h in enumerate(heads):
        if h['status']!='learnable':pred[:,j]=np.nan if h['mean'] is None else h['mean']
        elif j<2:pred[:,j]=torch.sigmoid(torch.from_numpy(raw[:,j])).numpy()
        elif j==2:pred[:,j]=np.clip(raw[:,j],0,1)
        else:pred[:,j]=np.maximum(raw[:,j],0)
    return pred*SCALES

def select(pred,eligible,active):
    remaining=[i for i in range(3) if eligible[i]]
    if not remaining:return None
    for j,tol in enumerate((.05,.05,.05,.02,1.)):
        if j not in active:continue
        if not np.isfinite(pred[remaining,j]).all():return None
        best=np.max(pred[remaining,j]) if j in (0,2) else np.min(pred[remaining,j])
        remaining=[i for i in remaining if pred[i,j]>=best-tol] if j in (0,2) else [i for i in remaining if pred[i,j]<=best+tol]
    return remaining[0]

def ridge(X,y,mask,groups,Xe,heads):
    pred=np.full((len(Xe),5),np.nan);raw=pred.copy();fits={}
    for j,h in enumerate(heads):
        if h['status']=='unsupported':continue
        if h['status']!='learnable':pred[:,j]=h['mean'];raw[:,j]=h['mean'];continue
        idx=mask[:,j];w=source_weights(idx,groups)[idx];x=X[idx,1024:1045].astype(float);z=y[idx,j]/SCALES[j]
        xm=np.average(x,weights=w,axis=0);ym=np.average(z,weights=w)
        u,s,vt=np.linalg.svd((x-xm)*np.sqrt(w[:,None]),full_matrices=False)
        coef=vt.T@((s/(s*s+1))* (u.T@((z-ym)*np.sqrt(w))))
        intercept=ym-xm@coef;values=Xe[:,1024:1045]@coef+intercept;raw[:,j]=values
        pred[:,j]=np.clip(values,0,1) if j<3 else np.maximum(values,0)
        fits[j]=dict(alpha=1.,coef=coef.tolist(),intercept=float(intercept),source_weight_sum=float(w.sum()))
    return pred*SCALES,raw*SCALES,fits
