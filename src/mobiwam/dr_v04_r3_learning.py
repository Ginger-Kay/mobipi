"""Fixed Source-weighted masked development recipe and train-only baselines."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SCALES=np.array([1.,1.,1.,2.,120.],dtype=np.float64)

def source_weights(mask, groups):
    w=np.zeros(len(mask),dtype=np.float64)
    for g in sorted(set(groups[mask])):
        idx=mask & (groups==g);w[idx]=1./idx.sum()
    return w

def support(y, mask, groups):
    result=[]
    for j in range(5):
        idx=mask[:,j];v=y[idx,j]/SCALES[j];w=source_weights(idx,groups)[idx]
        status="unsupported" if not len(v) else ("single_class" if j<2 and len(np.unique(v))==1 else ("constant" if j>=2 and np.ptp(v)<=1e-8 else "learnable"))
        result.append(dict(status=status,rows=int(idx.sum()),sources=len(set(groups[idx])),
                           mean=float(np.average(v,weights=w)) if len(v) else None,
                           positive=int((v==1).sum()) if j<2 else None,
                           negative=int((v==0).sum()) if j<2 else None))
    return result

def fit_scaler(X, groups):
    w=source_weights(np.ones(len(X),dtype=bool),groups);w=w/w.sum()
    mean=np.sum(X.astype(np.float64)*w[:,None],axis=0)
    std=np.sqrt(np.sum((X-mean)**2*w[:,None],axis=0))
    return mean,std

def scale(X, mean, std):
    out=np.zeros_like(X,dtype=np.float32);active=std>=1e-6
    out[:,active]=((X[:,active]-mean[active])/std[active]).astype(np.float32)
    return out

def masked_loss(raw, y, mask, groups, active):
    losses={}
    for j in active:
        idx=mask[:,j]
        if not bool(idx.any()):continue
        pred=raw[idx,j];target=y[idx,j]
        if j<2:values=F.binary_cross_entropy_with_logits(pred,target,reduction="none")
        else:
            pred=torch.sigmoid(pred) if j==2 else F.softplus(pred)
            values=F.smooth_l1_loss(pred,target,reduction="none",beta=.1)
        per_source=[]
        for g in torch.unique(groups[idx]):
            per_source.append(values[groups[idx]==g].mean())
        losses[j]=torch.stack(per_source).mean()
    # Graph-connected zero is safe for an all-missing batch.
    return (torch.stack(list(losses.values())).mean() if losses else raw.sum()*0),losses

class ActiveLinear(nn.Module):
    """Linear(1045,5) initialization with only learnable rows in optimizer."""
    def __init__(self,active):
        super().__init__()
        self.linear=nn.Linear(1045,5)
        self.linear.requires_grad_(False)
        self.register_buffer("active",torch.tensor(active,dtype=torch.long))
        self.weight=nn.Parameter(self.linear.weight.detach()[active].clone())
        self.bias=nn.Parameter(self.linear.bias.detach()[active].clone())
    def forward(self,x):
        return self.linear(x).index_copy(1,self.active,F.linear(x,self.weight,self.bias))
    def materialized(self):
        w=self.linear.weight.detach().clone();b=self.linear.bias.detach().clone()
        w[self.active]=self.weight.detach();b[self.active]=self.bias.detach()
        return dict(weight=w,bias=b)

def transform(raw, heads):
    p=np.empty_like(raw,dtype=np.float64)
    for j,h in enumerate(heads):
        if h["status"]!="learnable":p[:,j]=np.nan if h["mean"] is None else h["mean"]
        elif j<3:p[:,j]=torch.sigmoid(torch.from_numpy(raw[:,j])).numpy()
        else:p[:,j]=np.logaddexp(0.,raw[:,j])
    return p*SCALES

def baselines(X, y, mask, groups, tasks, routes, Xe, taskse, routese, heads):
    normalized=y/SCALES
    b1=np.full((len(Xe),5),np.nan);b2=b1.copy();b3=b1.copy();raw3=b1.copy();fits={}
    for j,h in enumerate(heads):
        if h["status"]=="unsupported":continue
        b1[:,j]=h["mean"];b2[:,j]=h["mean"]
        idx=mask[:,j];w=source_weights(idx,groups)[idx]
        for task in sorted(set(tasks)):
            for route in ("E","D","A"):
                cell=idx & (tasks==task) & (routes==route)
                if cell.any():
                    cw=source_weights(cell,groups)[cell]
                    b2[(taskse==task)&(routese==route),j]=np.average(normalized[cell,j],weights=cw)
        if h["status"]!="learnable":b3[:,j]=h["mean"];raw3[:,j]=h["mean"];continue
        x=X[idx,-21:].astype(np.float64);z=normalized[idx,j]
        xm=np.average(x,weights=w,axis=0);ym=np.average(z,weights=w)
        A=(x-xm)*np.sqrt(w[:,None]);target=(z-ym)*np.sqrt(w)
        u,s,vt=np.linalg.svd(A,full_matrices=False)
        coef=vt.T@((s/(s*s+1.))* (u.T@target))
        intercept=ym-xm@coef
        pred=Xe[:,-21:]@coef+intercept;raw3[:,j]=pred
        b3[:,j]=np.clip(pred,0,1) if j<3 else np.maximum(0,pred)
        fits[j]=dict(coef=coef.tolist(),intercept=float(intercept),alpha=1.,source_weight_sum=float(w.sum()))
    return dict(B1=b1*SCALES,B2=b2*SCALES,B3=b3*SCALES,B3_unclipped=raw3*SCALES),fits

def select(pred, active):
    if not active:return None
    remaining=list(range(3))
    for j,tol in enumerate((.05,.05,.05,.02,1.)):
        if j not in active:continue
        vals=pred[remaining,j]
        if not np.isfinite(vals).all():return None
        best=max(vals) if j in (0,2) else min(vals)
        remaining=[i for i in remaining if (pred[i,j]>=best-tol if j in (0,2) else pred[i,j]<=best+tol)]
    return remaining[0]
