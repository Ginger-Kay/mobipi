import numpy as np
from mobiwam.dr_v04_r2_pack import labels,LABELS

def test_unknown_label_differs_from_observed_false_and_zero():
    rows=[dict(group_id='a',machine_eligible_for_gate=True,**dict(zip(LABELS,(False,0.,False,0.,1.)))) for i in range(3)]
    rows[1]['failure']=None
    y,mask,route_ok,group_ok=labels(rows)
    assert y[0,0]==0 and y[0,2]==0 and mask[0].all()
    assert np.isnan(y[1,2]) and not mask[1,2]
    assert route_ok.tolist()==[True,False,True] and not group_ok.any()

def test_complete_pair_group_qualification_is_atomic():
    row=dict(group_id='a',machine_eligible_for_gate=True,success=True,progress=1.,failure=False,base_path_m=0.,completion_time_s=1.)
    assert labels([dict(row),dict(row),dict(row)])[3].all()
    assert not labels([dict(row),dict(row)])[3].any()
