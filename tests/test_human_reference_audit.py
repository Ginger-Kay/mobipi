import copy
import numpy as np
import pytest
from human_reference_audit import validate_human_a


def original():
    result=dict(route="A",steps=3,events=[
        dict(event="collection_begin",step=0,operator_id="operator"),
        dict(event="contact",step=1),
        dict(event="pause",step=2),dict(event="resume",step=2)])
    meta=dict(route="A",record_type="practice",version="human-scene-pilot-v1",operator_id="operator")
    return result,meta,np.zeros((3,12)),np.zeros((4,163))


def test_manual_markers_preserved_without_input_mutation():
    args=original();before=copy.deepcopy(args)
    assert validate_human_a(*args)==before[0]["events"]
    assert args[:2]==before[:2]
    assert np.array_equal(args[2],before[2]) and np.array_equal(args[3],before[3])


@pytest.mark.parametrize("mutate",[
    lambda r,m,a,s:r.update(route="D"),
    lambda r,m,a,s:m.update(record_type="primary"),
    lambda r,m,a,s:r["events"].append(dict(event="dock_settled_reobserve_feedback_reset",step=2)),
    lambda r,m,a,s:r["events"].append(dict(event="pause",step=1)),
    lambda r,m,a,s:r["events"].append(dict(event="resume",step=2)),
    lambda r,m,a,s:m.update(operator_id="other"),
    lambda r,m,a,s:r.update(steps=4),
    lambda r,m,a,s:a.__setitem__((0,0),np.nan),
])
def test_nonhuman_or_unknown_state_transition_fails_closed(mutate):
    args=original();mutate(*args)
    with pytest.raises(ValueError):
        validate_human_a(*args)
