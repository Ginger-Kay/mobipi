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

from human_reference_audit import validate_human_recording, phase_at

def human_d():
    r,m,a,s=original()
    r["route"]=m["route"]="D"
    dock=dict(event="docked_state",step=1,base_qpos=[0,0,0],base_qvel=[0,0,0])
    r["events"]=[r["events"][0],dict(event="docked",step=1),dock,dict(event="contact",step=2)]
    m.update(paired_protocol_version="human-eda-v2-stowed",human_selected_dock={k:v for k,v in dock.items() if k!="event"})
    a[0,6]=-1; a[1:,11]=-1
    return r,m,a,s

def test_human_d_phase_and_actions():
    args=human_d()
    assert validate_human_recording(*args)==args[0]["events"]
    assert [phase_at(args[0],i) for i in range(3)]==["navigate","manipulate","manipulate"]

@pytest.mark.parametrize("mutate",[
 lambda r,m,a,s:r["events"][2].update(step=2),
 lambda r,m,a,s:r["events"][2].update(base_qvel=[.02,0,0]),
 lambda r,m,a,s:a.__setitem__((0,6),1),
 lambda r,m,a,s:a.__setitem__((0,1),.1),
 lambda r,m,a,s:a.__setitem__((1,7),.1),
 lambda r,m,a,s:r["events"].append(dict(event="controller_reset",step=2)),
 lambda r,m,a,s:r["events"].append(dict(event="docked",step=2)),
 lambda r,m,a,s:m.update(paired_protocol_version="unknown"),
])
def test_bad_human_d_fails_closed(mutate):
    args=human_d();mutate(*args)
    with pytest.raises(ValueError):validate_human_recording(*args)

def test_d_failure_before_dock_retains_navigation_only():
    r,m,a,s=human_d()
    r["events"]=r["events"][:1]
    m["human_selected_dock"]=None
    a[:]=0;a[:,6]=-1
    validate_human_recording(r,m,a,s)
    assert all(phase_at(r,i)=="navigate" for i in range(3))

def test_e_admits_fixed_base_and_rejects_motion():
    r,m,a,s=original();r["route"]=m["route"]="E"
    m["paired_protocol_version"]="human-eda-v2-stowed"
    a[:,11]=-1
    validate_human_recording(r,m,a,s)
    a[2,8]=.1
    with pytest.raises(ValueError,match="base input"):validate_human_recording(r,m,a,s)

def test_primary_cannot_be_admitted_without_freeze():
    r,m,a,s=original();m["record_type"]="primary"
    with pytest.raises((KeyError,ValueError)):validate_human_recording(r,m,a,s)

from human_reference_audit import load_partial_tail
import json

def test_missing_safety_tail_fails_closed(tmp_path):
    with pytest.raises(ValueError,match="missing"):
        load_partial_tail(tmp_path,dict(reason="native_forbidden_contact_stop",steps=3))

def test_partial_tail_boundary_and_action_integrity(tmp_path):
    result=dict(reason="native_forbidden_contact_stop",steps=3)
    stop=dict(step=3,failure=dict(step=3,kind="native_forbidden_contact"))
    (tmp_path/"safety-stop.json").write_text(json.dumps(stop))
    np.savez(tmp_path/"partial-control-step.npz",initial_integration=np.zeros(10),terminal_integration=np.ones(10),attempted_action=np.zeros(12))
    assert load_partial_tail(tmp_path,result)["record"]==stop
    stop["step"]=2;(tmp_path/"safety-stop.json").write_text(json.dumps(stop))
    with pytest.raises(ValueError,match="boundary"):load_partial_tail(tmp_path,result)


def supplement(tmp_path):
    from test_human_primary import frozen
    from pathlib import Path
    cfg=frozen(tmp_path)
    cfg['paired_protocol_version']='human-eda-v2-stowed'
    receipt_path=Path(cfg['freeze_receipt'])
    receipt=json.loads(receipt_path.read_text());receipt['config']=cfg
    receipt_path.write_text(json.dumps(receipt))
    r,m,a,s=original()
    r['source']=cfg['source'];r['events'][0]['record_type']='reference_supplement'
    m.update(record_type='reference_supplement',freeze_receipt=cfg['freeze_receipt'],
             config_version=cfg['config_version'],source_id=Path(cfg['source']).name,
             paired_protocol_version=cfg['paired_protocol_version'])
    return r,m,a,s


def test_reference_supplement_keeps_record_classification_and_original_inputs(tmp_path):
    args=supplement(tmp_path);before=copy.deepcopy(args)
    assert validate_human_recording(*args)==before[0]['events']
    assert args[:2]==before[:2]
    assert args[1]['record_type']=='reference_supplement'
    assert np.array_equal(args[2],before[2]) and np.array_equal(args[3],before[3])


@pytest.mark.parametrize('bad_field',['source','event_classification','frozen_source'])
def test_reference_supplement_rejects_identity_or_freeze_changes(tmp_path,bad_field):
    from pathlib import Path
    args=supplement(tmp_path);r,m,a,s=args
    if bad_field=='source':r['source']='/wrong-source'
    elif bad_field=='event_classification':r['events'][0]['record_type']='primary'
    else:(Path(r['source'])/'integration.npy').write_text('changed')
    with pytest.raises(ValueError):validate_human_recording(*args)
