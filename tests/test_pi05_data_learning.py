import numpy as np
from mobiwam.pi05_data_learning import hierarchy_weights,fit_scaler,SCALES,support,choose,labels,decode

def test_parent_config_route_weight_invariance_to_config_and_episode_counts():
    p=np.array(['p1']*6+['p2']*2);c=np.array(['a']*2+['b']*4+['c']*2);r=np.array(['E','A','E','E','D','A','E','A'])
    mask=np.ones((8,5),bool);mask[1,1]=False;mask[2:6,2]=False
    w=hierarchy_weights(p,c,r,mask)
    assert np.allclose(w.sum(0),1.)
    assert np.allclose(w[p=='p1'].sum(0),.5)
    assert np.allclose(w[p=='p2'].sum(0),.5)
    assert np.isclose(w[2,0]+w[3,0],w[4,0])
    assert np.isclose(w[0,2]+w[1,2],.5)

def test_train_scaler_hierarchy_and_constant_dimensions():
    X=np.array([[0.,7.],[0.,7.],[8.,7.],[8.,7.],[8.,7.]])
    p=['p1','p1','p2','p2','p2'];c=['a','a','b','b','b'];r=['E','A','E','D','A']
    mean,std=fit_scaler(X,p,c,r,np.ones(5,bool))
    assert np.allclose(mean,[4.,7.]);assert np.allclose(std,[4.,0.])

def test_time_300_joint_stop_not_collision_and_constant_collision_not_ranked():
    assert SCALES[4]==300.
    actual,mask=labels({'status':'joint_margin_stop','usable_scientific_outcome':True,'native_success':False,
                       'terminal_native_opening':.2,'actual_base_path_m':.4,'terminal_duration_s':81.}, {'native_collision':False})
    assert np.all(mask) and actual[0]==0 and actual[1]==0 and np.isclose(actual[2],.8)
    y=np.array([[0.,0.,.2,1.,50.],[1.,0.,1.,.5,20.]])
    heads=support(y,np.ones_like(y,bool));assert heads[1]['status']=='constant' and not heads[1]['ranking_supported']
    preds={'E':np.array([.8,.99,.8,1.,100.]),'A':np.array([.8,0.,.8,1.,100.])}
    assert choose(preds,{'E':True,'A':True},heads)=='E'
    assert choose(preds,{},heads)=='X'
    physical=decode(np.array([[0.,0.,1.2,-.3,1.]]),[{'status':'learnable'}]*5)
    assert np.allclose(physical,[[.5,.5,1.,0.,300.]])

def test_masks_do_not_make_unknowns_into_negative_labels():
    y=np.array([[0.,np.nan,.2,1.,50.],[1.,np.nan,1.,.5,20.]])
    mask=np.isfinite(y);heads=support(y,mask)
    assert heads[1]['status']=='unknown';assert np.all(hierarchy_weights(['a','b'],['a','b'],['E','A'],mask)[:,1]==0)
    actual,mask=labels({'status':'engineering_unknown','usable_scientific_outcome':False,'native_success':False},{'native_collision':False})
    assert not np.any(mask) and np.isnan(actual).all()
    actual,mask=labels({'status':'X_no_legal_candidate','usable_scientific_outcome':True,'native_success':False},{})
    assert not np.any(mask) and np.isnan(actual).all()
