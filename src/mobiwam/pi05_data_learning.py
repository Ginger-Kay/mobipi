"""Fixed PI05-DATA-v1 units, masked hierarchy, and physical-unit selection."""
from collections import defaultdict
import numpy as np
import torch
from torch import nn

HEADS=('success','collision','progress','path','terminal_duration')
SCALES=np.array([1.,1.,1.,2.,300.],np.float32)

def labels(receipt,audit):
    """Observed native termination labels; collision uses actual evidence only."""
    y=np.full(5,np.nan,np.float32);mask=np.zeros(5,bool)
    status=receipt.get('status','engineering_unknown')
    terminal=bool(receipt.get('usable_scientific_outcome')) and not status.startswith(('X_','engineering_'))
    if terminal:
        y[0]=float(receipt['native_success']);mask[0]=True
        opening=receipt.get('terminal_native_opening')
        if opening is not None and np.isfinite(opening):y[2]=np.clip(1-opening,0,1);mask[2]=True
        for j,key in [(3,'actual_base_path_m'),(4,'terminal_duration_s')]:
            value=receipt.get(key)
            if value is not None and np.isfinite(value):y[j]=value;mask[j]=True
    collision=audit.get('native_collision')
    if isinstance(collision,(bool,np.bool_)) and (terminal or collision):y[1]=float(collision);mask[1]=True
    return y,mask

def hierarchy_weights(parents,configs,routes,mask):
    """Equal parents -> supervised configs -> supervised routes -> repetitions."""
    parents=np.asarray(parents);configs=np.asarray(configs);routes=np.asarray(routes);mask=np.asarray(mask,bool)
    if mask.ndim==1:mask=mask[:,None]
    weights=np.zeros(mask.shape,np.float64)
    for head in range(mask.shape[1]):
        grouped=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        for i in np.flatnonzero(mask[:,head]):grouped[parents[i]][configs[i]][routes[i]].append(i)
        for p,cs in grouped.items():
            for c,rs in cs.items():
                for route,indices in rs.items():
                    weights[indices,head]=1/(len(grouped)*len(cs)*len(rs)*len(indices))
    return weights.astype(np.float32)

def fit_scaler(X,parents,configs,routes,eligible):
    X=np.asarray(X,np.float64);w=hierarchy_weights(parents,configs,routes,eligible)[:,0].astype(np.float64)
    if not np.any(w):raise ValueError('no supervised train-parent eligible candidate inputs')
    w/=w.sum();mean=(X*w[:,None]).sum(0);var=((X-mean)**2*w[:,None]).sum(0)
    std=np.sqrt(np.maximum(var,0));std[std<1e-6]=0.
    return mean.astype(np.float32),std.astype(np.float32)

def scale(X,mean,std):
    return np.where(std>=1e-6,(np.asarray(X)-mean)/np.where(std>=1e-6,std,1),0).astype(np.float32)

def support(y,mask):
    out=[]
    for j,name in enumerate(HEADS):
        vals=np.asarray(y)[np.asarray(mask)[:,j],j]
        if not len(vals):out.append(dict(name=name,status='unknown',count=0,constant=None));continue
        constant=float(vals.mean()) if np.max(vals)-np.min(vals)<=1e-8 else None
        out.append(dict(name=name,status='constant' if constant is not None else 'learnable',count=len(vals),constant=constant,
            ranking_supported=not(name=='collision' and len(np.unique(vals))<2),single_class=len(np.unique(vals))<2))
    return out

class Head(nn.Module):
    def __init__(self,kind):
        super().__init__();self.net=nn.Sequential(nn.Linear(1048,32),nn.ReLU(),nn.Linear(32,5)) if kind=='MLP' else nn.Linear(1048,5)
    def forward(self,x):return self.net(x)

def masked_loss(pred,target,weights,active):
    if not active:return pred.sum()*0,{}
    terms={}
    for j in active:
        if j<2:loss=torch.nn.functional.binary_cross_entropy_with_logits(pred[:,j],target[:,j],reduction='none')
        else:loss=(pred[:,j]-target[:,j])**2
        terms[j]=(loss*weights[:,j]).sum()
    return torch.stack(list(terms.values())).mean(),terms

def decode(raw,heads):
    result=np.array(raw,dtype=np.float64,copy=True)
    result[:,:2]=1/(1+np.exp(-np.clip(result[:,:2],-80,80)));result*=SCALES
    for j,h in enumerate(heads):
        if h['status']=='constant':result[:,j]=h['constant']
        if h['status']=='unknown':result[:,j]=np.nan
    return result

def choose(predictions,valid,heads):
    """No supported candidate -> X; all tie thresholds in physical units."""
    pool=[r for r in 'EDA' if valid.get(r,False)]
    if not pool:return 'X'
    collision_rank=heads[1].get('ranking_supported',False)
    for j,window,maximum in [(0,.05,True),(1,.05,False),(2,.05,True),(3,.02,False),(4,1.,False)]:
        if j==1 and not collision_rank:continue
        values=[predictions[r][j] for r in pool]
        if not all(np.isfinite(v) for v in values):continue
        best=max(values) if maximum else min(values)
        pool=[r for r in pool if (predictions[r][j]>=best-window if maximum else predictions[r][j]<=best+window)]
    return min(pool,key='EDA'.index)
