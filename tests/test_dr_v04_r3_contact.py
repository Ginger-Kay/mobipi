import numpy as np
import mujoco
import pytest
from types import SimpleNamespace
from unittest.mock import patch
from mobiwam.contact_rules import exempt_finger_pad_pair, allowed_contact, FINGER_PAD_PAIR
from mobiwam.reference_collision import SweptGeometry
from mobiwam.reference_formal_substep import FormalSubstepMonitor, FormalSafetyStop

@pytest.mark.parametrize("phase",["precontact","navigate","stow","manipulate"])
def test_exact_symmetric_pair_only(phase):
    a,b=sorted(FINGER_PAD_PAIR)
    assert allowed_contact(a,b,phase,"target") and allowed_contact(b,a,phase,"target")
    for other in (None,"",a,a+"_extra","gripper1_right_finger2_pad_collision","target_handle","robot0_arm"):
        assert not exempt_finger_pad_pair(a,other)

def test_physical_contact_remains_but_monitor_and_sweep_exempt_it():
    a,b=sorted(FINGER_PAD_PAIR)
    m=mujoco.MjModel.from_xml_string(f'''<mujoco><option gravity="0 0 0"/><worldbody>
      <body pos="0 0 0"><joint type="slide" axis="1 0 0"/><geom name="{a}" size=".05" mass="1"/></body>
      <body pos=".08 0 0"><joint type="slide" axis="1 0 0"/><geom name="{b}" size=".05" mass="1"/></body>
    </worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    assert d.ncon>0
    before={k:getattr(m,k).copy() for k in ("geom_contype","geom_conaffinity","geom_friction","geom_solref","geom_solimp")}
    check=SweptGeometry(m)
    assert check.path([d.qpos.copy(),d.qpos.copy()],["navigate"])["valid"]
    ref=SimpleNamespace(model_data=lambda:(m,d),env=SimpleNamespace(sim=SimpleNamespace(step=lambda:None),lite_physics=False))
    monitor=FormalSubstepMonitor(ref,"target");monitor.set_boundary(0,"navigate")
    assert not list(monitor.forbidden_contacts()) and monitor.exempt_contacts
    for k,v in before.items():np.testing.assert_array_equal(v,getattr(m,k))
    # A different pair remains in the collision gate.
    xml='''<mujoco><worldbody><geom name="wall" size=".05"/>
     <body><joint type="slide"/><geom name="gripper0_other_pad" size=".05" mass="1"/></body>
    </worldbody></mujoco>'''
    other=SweptGeometry(mujoco.MjModel.from_xml_string(xml))
    assert not other.path([[0],[0]],["navigate"])["valid"]
