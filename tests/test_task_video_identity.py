"""Real recorder integration on an explicitly supplied independent dev Source.

No actions or outcomes: native stills and original HDF5 initial states only.
"""
import argparse,json,os,sys
from pathlib import Path
from unittest.mock import patch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from teleop_reference import Reference
from mobiwam.task_video_identity import (IdentityHold,observe_native,validate_recording,
                                        human_delivery,sha)

@pytest.fixture(scope='module')
def recorded():
    source=os.environ.get('OBC_IDENTITY_DEV_SOURCE');output=os.environ.get('OBC_IDENTITY_TEST_OUTPUT')
    if not source or not output:pytest.skip('explicit independent dev Source/output required')
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    ref=Reference(argparse.Namespace(output=str(out/'env'),task='CloseDrawer',layout=0,style=0,seed=7,self_test=True,source=str(source),replay_attempt=None,resume_attempt=None,width=640,height=480))
    def forbidden(*args,**kwargs):raise AssertionError('env.step forbidden in identity regression')
    ref.env.step=forbidden
    expected=dict(task='CloseDrawer',fixture_name=ref.env.drawer.name,fixture_class='Drawer',model_sha256=sha(source/'model.xml'))
    context=dict(run_id='identity-regression',group_id='independent-dev-drawer17')
    try:
        ref.route='E';e=ref.identity_preview(out/'E'/'zero-action-E',expected,context)
        ref.route='A';a=ref.identity_preview(out/'A'/'zero-action-A',expected,context,camera=dict(lookat=[3.8,-1.,.65],distance=2.4,azimuth=90.,elevation=-15.))
        yield ref,expected,e,a,out
    finally:
        if ref.renderer:ref.renderer.close()
        if ref.observation_renderer:ref.observation_renderer.close()
        ref.env.close()

def test_native_and_recorder_pair_accept(recorded):
    ref,expected,e,a,out=recorded
    assert observe_native(ref,expected)['fixture_class']=='Drawer'
    v=validate_recording(e,dict(route='E',attempt_id='zero-action-E'))
    assert v['steps']==0 and len(v['decoded_frames_sha256'])==1
    assert human_delivery(e,out/'extra-human-review')['formal_qualification'] is False

def test_e_a_manifest_exchange_reject(recorded):
    _,_,e,a,_=recorded
    with pytest.raises(IdentityHold,match='wrong route'):validate_recording(a,dict(route='E'))

def test_swapped_video_reject(recorded):
    _,_,e,a,_=recorded
    ep=e.parent/'original.mp4';original=ep.read_bytes()
    try:
        ep.write_bytes((a.parent/'original.mp4').read_bytes())
        with pytest.raises(IdentityHold,match='wrong video/attempt'):validate_recording(e)
    finally:ep.write_bytes(original)

def test_cross_attempt_hdf5_reject(recorded):
    _,_,e,a,_=recorded
    ep=e.parent/'demo.hdf5';original=ep.read_bytes()
    try:
        ep.write_bytes((a.parent/'demo.hdf5').read_bytes())
        with pytest.raises(IdentityHold,match='wrong video/attempt'):validate_recording(e)
    finally:ep.write_bytes(original)

def test_declared_task_actual_target_reject(recorded):
    ref,expected,*_=recorded
    with pytest.raises(IdentityHold,match='declared task'):observe_native(ref,dict(expected,task='PushChair'))
    with pytest.raises(IdentityHold,match='actual target object'):observe_native(ref,dict(expected,fixture_name='stool_1_island_group'))

def test_wrong_checker_reject(recorded):
    ref,expected,*_=recorded
    with patch.object(ref.env,'_check_success',lambda:True):
        with pytest.raises(IdentityHold,match='checker'):observe_native(ref,expected)

def test_missing_fields_reject(recorded):
    ref,expected,e,_,out=recorded
    with pytest.raises(IdentityHold,match='missing declared'):observe_native(ref,{})
    old=e.read_text();v=json.loads(old);del v['binding']['native']['checker']
    try:
        e.write_text(json.dumps(v))
        with pytest.raises(IdentityHold,match='missing native checker'):human_delivery(e,out/'rejected-review')
        assert not (out/'rejected-review').exists()
    finally:e.write_text(old)

def test_actual_audit_rejects_legacy_without_manifest(recorded):
    from dr_v04_formal_audit import audit
    with pytest.raises(IdentityHold,match='missing recorder identity manifest'):
        audit({},'legacy','E',recorded[-1]/'missing-legacy-attempt')
