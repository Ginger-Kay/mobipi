import json
import numpy as np
import pytest
from mobiwam.reference_terminal_step import terminal_record


def fixture(tmp_path):
    np.savez(tmp_path/'partial-control-step.npz',initial_integration=np.zeros(4),terminal_integration=np.ones(4),attempted_action=np.zeros(12),completed_physics_substeps=3)
    np.savez(tmp_path/'formal-native-substeps.npz',qpos=np.zeros((4,2)),sim_time=np.arange(4),step_index=np.array([0,0,1]))
    (tmp_path/'formal-substep-stop.json').write_text(json.dumps(dict(step=1,substep=2,phase='manipulate')))
    return dict(steps=1,reason='native_forbidden_contact_stop'),[dict(step=0,stage='manipulate'),dict(step=1,stage='manipulate')]


def test_terminal_boundary(tmp_path):
    result,feedback=fixture(tmp_path);r=terminal_record(tmp_path,result,feedback)
    assert r['prefix_substeps']==2 and r['tail_qpos'].shape==(2,2)


def test_missing_extra_feedback_rejected(tmp_path):
    result,feedback=fixture(tmp_path)
    with pytest.raises(ValueError):terminal_record(tmp_path,result,feedback[:-1])


def test_nonstop_extra_feedback_rejected(tmp_path):
    result,feedback=fixture(tmp_path);result['reason']='tracking_stall'
    with pytest.raises(ValueError):terminal_record(tmp_path,result,feedback)


def test_mistimed_stop_rejected(tmp_path):
    result,feedback=fixture(tmp_path);feedback[-1]['step']=0
    with pytest.raises(ValueError):terminal_record(tmp_path,result,feedback)


def test_wrong_substep_count_rejected(tmp_path):
    result,feedback=fixture(tmp_path)
    np.savez(tmp_path/'partial-control-step.npz',initial_integration=np.zeros(4),terminal_integration=np.ones(4),attempted_action=np.zeros(12),completed_physics_substeps=4)
    with pytest.raises(ValueError):terminal_record(tmp_path,result,feedback)


def test_completed_route_unchanged(tmp_path):
    assert terminal_record(tmp_path,dict(steps=2,reason='tracking_stall'),[{},{}]) is None
