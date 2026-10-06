import numpy as np
import pytest
import torch
from mobiwam.sim_sprint_learning import masked_loss,require_group_split,select,fit_scaler,scale

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
