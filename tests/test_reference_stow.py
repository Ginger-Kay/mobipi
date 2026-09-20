import copy
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from reference_stow import build_stow, load_stow


def fixture():
    links=''
    for i in range(7):
        links+='<body><joint type="hinge" axis="0 0 1" range="-2 2"/><geom type="sphere" size=".02" mass="1"/>'
    links+='<site name="eef" pos="0 0 .2"/>'+'</body>'*7
    model=mujoco.MjModel.from_xml_string('<mujoco><compiler angle="radian"/><worldbody><body name="base">'+links+'</body></worldbody></mujoco>')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    return NS(model_data=lambda:(model,data),base_body='base',
        robot=NS(part_controllers={'right':NS(qpos_index=np.arange(7),qvel_index=np.arange(7))},eef_site_id={'right':0}))


def test_build_load_preserves_source():
    ref=fixture();m,d=ref.model_data();before=d.qpos.copy()
    record=build_stow(ref)
    pos,rot,q=load_stow(ref,record)
    np.testing.assert_array_equal(d.qpos,before)
    np.testing.assert_allclose(pos,[0,0,.2])
    assert np.min(m.jnt_range[:,1]-q)>.015


@pytest.mark.parametrize('field,value', [('version','old'),('source_qpos_sha256','wrong'),
    ('solver_margin_rad',.015),('local_pos',[1,0,0]),('local_rot',np.zeros((3,3)).tolist()),
    ('arm_qpos',[float('nan')]*7),('arm_qpos',[1.99]*7)])
def test_bad_or_unbound_records_rejected(field,value):
    ref=fixture();record=copy.deepcopy(build_stow(ref));record[field]=value
    with pytest.raises(ValueError):load_stow(ref,record)


def test_changed_source_rejected():
    ref=fixture();record=build_stow(ref)
    ref.model_data()[1].qpos[0]=.1
    with pytest.raises(ValueError,match='source mismatch'):load_stow(ref,record)
