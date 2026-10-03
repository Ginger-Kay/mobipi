import numpy as np
import torch
from mobiwam.dr_v04_r3_learning import *

def test_mask_source_equal_gradient_and_unknown_no_nan():
    raw=torch.zeros(4,5,requires_grad=True)
    y=torch.full((4,5),float("nan"));mask=torch.zeros(4,5,dtype=torch.bool)
    y[:3,0]=1.;y[3,0]=0.;mask[:,0]=True
    loss,heads=masked_loss(raw,y,mask,torch.tensor([0,0,0,1]),[0,1])
    loss.backward()
    np.testing.assert_allclose(raw.grad[:,0],[-1/12,-1/12,-1/12,1/4],rtol=1e-6)
    assert torch.isfinite(loss) and torch.count_nonzero(raw.grad[:,1:])==0
    empty,_=masked_loss(raw,y,torch.zeros_like(mask),torch.tensor([0,0,0,1]),[0])
    assert empty.item()==0

def test_head_support_scaler_and_route_order():
    y=np.array([[1,0,.2,0.,10],[0,0,.8,1.,30],[1,0,.5,.3,20]],float)
    mask=np.ones((3,5),bool);groups=np.array(["g0","g0","g1"])
    h=support(y,mask,groups)
    assert h[1]["status"]=="single_class"
    assert h[0]["mean"]==.75
    X=np.array([[0,1],[2,1],[10,1]],float)
    mean,std=fit_scaler(X,groups);np.testing.assert_allclose(mean,[5.5,1])
    assert scale(X,mean,std)[:,1].sum()==0
    p=np.array([[.9,0,.5,.2,30],[.88,0,.7,.3,10],[.2,0,.9,.1,1]])
    assert select(p,[0,2,3,4])==1
    assert select(p,[]) is None

def test_constant_rows_have_no_decay_and_active_matches_full_linear_initialization():
    torch.manual_seed(17);expected=torch.nn.Linear(1045,5)
    torch.manual_seed(17);model=ActiveLinear([0,2])
    torch.testing.assert_close(model.materialized()["weight"],expected.weight)
    before=model.materialized()
    opt=torch.optim.AdamW([dict(params=[model.weight],weight_decay=.05),dict(params=[model.bias],weight_decay=0.)],lr=3e-4)
    loss=model(torch.ones(2,1045))[:,[0,2]].sum();loss.backward();opt.step()
    after=model.materialized()
    torch.testing.assert_close(after["weight"][[1,3,4]],before["weight"][[1,3,4]],rtol=0,atol=0)
    assert not torch.equal(after["weight"][0],before["weight"][0])

def test_fixed_geometry_ridge_uses_unpenalized_intercept_and_train_only_cells():
    X=np.zeros((3,1045));X[:,-1]=[-1,0,1]
    y=np.tile([1,0,.5,2.,120.],(3,1)).astype(float);y[:,2]=[0,.5,1]
    mask=np.ones((3,5),bool);groups=np.array(["a","b","c"])
    tasks=np.array(["T"]*3);routes=np.array(["E"]*3)
    h=support(y,mask,groups)
    out,fit=baselines(X,y,mask,groups,tasks,routes,h and X,np.array(["T"]*3),np.array(["D"]*3),h)
    np.testing.assert_allclose(out["B2"][:,2],.5)
    np.testing.assert_allclose(out["B3"][:,2],[1/6,.5,5/6])
    assert fit[2]["alpha"]==1 and fit[2]["intercept"]==.5
