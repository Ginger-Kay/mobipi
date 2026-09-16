import mujoco
import numpy as np
import pytest
from mobiwam.replay_diagnostics import state_fields, summarize_drift


def test_joint_columns_and_action_index_locate_drift():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <freejoint name="object"/><geom size=".1" mass="1"/>
      <body><joint name="hinge"/><geom size=".1" mass="1"/></body>
      </body></worldbody></mujoco>''')
    fields=state_fields(model)
    assert len(fields)==1+model.nq+model.nv+model.na
    column=next(i for i,f in enumerate(fields) if f['name']=='hinge' and f['component']=='qvel')
    expected=np.zeros((3,len(fields)));actual=expected.copy();actual[2,column]=.2
    receipt=summarize_drift(expected,actual,fields)
    row=receipt['fields'][column]
    assert row['max_abs_error']==.2
    assert row['first_gt_threshold_after_action']==1
    assert row['max_error_after_action']==1
    assert receipt['fields'][0]['first_gt_threshold_after_action'] is None


def test_bad_alignment_and_nonfinite_states_are_rejected():
    fields=[dict(component='time',name='time',coordinate=0)]
    with pytest.raises(ValueError):summarize_drift([[0]],[[0],[1]],fields)
    with pytest.raises(ValueError):summarize_drift([[0]],[[float('nan')]],fields)
