import numpy as np
from pi05_drawer_obc_fit_v2 import weights, select


def test_half_config_weights_and_head_mask_renormalization():
    rows=[dict(parent=p,config_id=f'{p}-{c}',offset=c,route=t) for p in [114,151] for c in ['original','near05'] for t in 'EDA']
    mask=np.ones((12,2),bool);mask[0:3,1]=False;mask[8,1]=False
    w=weights(rows,mask)
    assert np.allclose(w.sum(0),1)
    assert np.isclose(w[:3,0].sum(),1/6)
    assert np.isclose(w[3:6,0].sum(),1/3)
    assert np.all(w[:3,1]==0)
    assert np.isclose(w[3:6,1].sum(),.5)
    assert w[8,1]==0
    assert np.isclose(w[6:8,1].sum(),1/6)
    assert np.allclose(weights(rows,mask,half=False).sum(0),1)


def test_selector_success_window_sequential_cost_and_collision_unsupported():
    heads=[dict(status='learnable',ranking_supported=True) for _ in range(5)]
    p={'E':np.array([.8,np.nan,.9,.2,10]),'D':np.array([.77,np.nan,.9,.1,20]),'A':np.array([.70,np.nan,1.,0.,0.])}
    assert select(p,dict.fromkeys('EDA',True),heads)['route']=='D'
    p['D'][0]=np.nan
    assert select(p,dict.fromkeys('EDA',True),heads)['route']=='E'
    for a in p.values():a[0]=np.nan
    result=select(p,dict.fromkeys('EDA',True),heads)
    assert result['route']=='E' and result['selector_fallback']
    assert select(p,dict.fromkeys('EDA',False),heads)['route'] is None
