import numpy as np
import pytest
import torch
from mobiwam.sim_sprint_learning import masked_loss,require_group_split,select,fit_scaler,scale
from mobiwam.sim_sprint_learning import restore_fit
import copy,random

def test_unequal_routes_have_equal_source_weight_and_missing_gradient():
    raw=torch.zeros((4,5),requires_grad=True);y=torch.zeros((4,5));y[:3,2]=1.;y[3,2]=3.
    mask=torch.zeros((4,5),dtype=torch.bool);mask[:,2]=True;mask[1,2]=False
    loss,_=masked_loss(raw,y,mask,torch.tensor([0,0,0,1]),[2]);assert float(loss)==5.
    loss.backward();assert raw.grad[1,2]==0 and raw.grad[3,2]==-3.

def test_scaler_equal_sources_and_small_std_zero():
    x=np.array([[0.,2.],[0.,2.],[6.,2.]])
    mean,std=fit_scaler(x,np.array(['a','a','b']));assert mean[0]==3.
    assert np.all(scale(x,mean,std)[:,1]==0)

def test_split_overlap_is_rejected_and_hard_invalid_cannot_win():
    with pytest.raises(ValueError):require_group_split(['family01'],['family01'])
    p=np.array([[1,0,1,0,0],[.8,0,.8,1,1],[.79,0,.8,.5,1]])
    assert select(p,[False,True,True],[0,2,3,4])==2
    assert select(p,[False]*3,[0,2,3,4]) is None

def test_optimizer_scheduler_and_rng_resume_equal_uninterrupted_toy_fit():
    torch.manual_seed(17)
    a=torch.nn.Linear(2,1);oa=torch.optim.AdamW(a.parameters(),lr=.001)
    sa=torch.optim.lr_scheduler.CosineAnnealingLR(oa,T_max=5)
    def step(model,opt,sched):
        x=torch.randn(3,2);opt.zero_grad();((model(x)-x.sum(1,keepdim=True))**2).mean().backward();opt.step();sched.step()
    for _ in range(3):step(a,oa,sa)
    cp=copy.deepcopy(dict(step=3,model=a.state_dict(),optimizer=oa.state_dict(),scheduler=sa.state_dict(),
        rng=dict(torch=torch.get_rng_state(),cuda=[],numpy=np.random.get_state(),python=random.getstate())))
    for _ in range(2):step(a,oa,sa)
    b=torch.nn.Linear(2,1);ob=torch.optim.AdamW(b.parameters(),lr=.001);sb=torch.optim.lr_scheduler.CosineAnnealingLR(ob,T_max=5)
    assert restore_fit(b,ob,sb,cp)==3
    for _ in range(2):step(b,ob,sb)
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters())) and sb.last_epoch==sa.last_epoch==5
