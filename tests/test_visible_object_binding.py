"""Native model mutation tests, plus the real recorder's zero-action entry."""
import os,json,copy,argparse,sys
from pathlib import Path
import pytest,mujoco,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mobiwam.visible_object_binding import (geometry_inventory,validate_native_geometry,
    VisibleBindingHold,model_geometry_sha,validate_render_model)

@pytest.fixture(scope='module')
def native():
    path=os.environ.get('OBC_VISIBLE_NATIVE_XML')
    if not path:pytest.skip('explicit diagnostic native XML required')
    m=mujoco.MjModel.from_xml_path(path);baseline=geometry_inventory(m)
    return m,baseline

def test_actual_compiled_pair_accepts(native):
    m,b=native;assert validate_native_geometry(m,b)
    assert len(b['geoms'])==m.ngeom
    assert b['geoms'][1244]['compiled_mesh_sha256']

def test_visible_chair_mesh_wrong_body_rejected(native):
    m,b=native;changed=copy.deepcopy(m);changed.geom_bodyid[1244]=228
    with pytest.raises(VisibleBindingHold,match='binding changed'):validate_native_geometry(changed,b)

def test_actual_shape_replaced_rejected(native):
    m,b=native;changed=copy.deepcopy(m)
    mesh=int(changed.geom_dataid[1244]);changed.mesh_vert[changed.mesh_vertadr[mesh],0]+=.03
    with pytest.raises(VisibleBindingHold,match='binding changed'):validate_native_geometry(changed,b)

def test_collision_owner_filter_and_parent_transform_rejected(native):
    m,b=native
    for field,index,value in [('geom_bodyid',1230,236),('geom_contype',1230,0),('body_pos',236,[0,0,0])]:
        changed=copy.deepcopy(m);getattr(changed,field)[index]=value
        with pytest.raises(VisibleBindingHold,match='binding changed'):validate_native_geometry(changed,b)

def test_stale_actual_model_id_rejected(native):
    m,b=native
    with pytest.raises(VisibleBindingHold,match='stale cached geom ID'):
        validate_native_geometry(m,b,dict(geom_id=1244,geom_name='island_cab_left_island_group_4_door_handle_handle'),'island_cab_left_island_group_4_slidejoint')

def test_checker_contact_target_mismatch_rejected(native):
    m,b=native
    with pytest.raises(VisibleBindingHold,match='checker joint is separate'):
        validate_native_geometry(m,b,dict(geom_id=1244,geom_name='stool_1_island_group_g0'),'island_cab_left_island_group_4_slidejoint')
    assert validate_native_geometry(m,b,dict(geom_id=1230,geom_name='island_cab_left_island_group_4_door_handle_handle'),'island_cab_left_island_group_4_slidejoint')

def test_kinematic_parent_effect_follows_actual_geometry(native):
    m,b=native;d=mujoco.MjData(m);mujoco.mj_forward(m,d);before=d.geom_xpos.copy();d.qpos[54]-=.1;mujoco.mj_forward(m,d)
    assert np.linalg.norm(d.geom_xpos[1221]-before[1221])==pytest.approx(.1)
    assert np.max(np.abs(d.geom_xpos[[1244,1247,1250]]-before[[1244,1247,1250]]))==0

@pytest.fixture(scope='module')
def recorder():
    from teleop_reference import Reference
    from mobiwam.task_video_identity import sha
    source=os.environ.get('OBC_VISIBLE_DEV_SOURCE');root=os.environ.get('OBC_VISIBLE_RECORDER_OUTPUT')
    if not source or not root:pytest.skip('explicit independent dev diagnostic required')
    root=Path(root);source=Path(source);root.mkdir(parents=True,exist_ok=False)
    ref=Reference(argparse.Namespace(output=str(root/'env'),task='CloseDrawer',layout=0,style=0,seed=7,self_test=True,source=str(source),replay_attempt=None,resume_attempt=None,width=640,height=480))
    def forbidden(*a,**kw):raise AssertionError('env.step forbidden in visible binding regression')
    ref.env.step=forbidden
    try:
        ref.route='E'
        expected=dict(task='CloseDrawer',fixture_name=ref.env.drawer.name,fixture_class='Drawer',model_sha256=sha(source/'model.xml'))
        manifest=ref.identity_preview(root/'zero-action-preview',expected,dict(run_id='visible-regression',group_id='independent-dev-drawer17'))
        yield ref,json.loads(manifest.read_text())['binding']
    finally:
        ref.recording=None
        if ref.renderer:ref.renderer.close()
        if ref.observation_renderer:ref.observation_renderer.close()
        ref.env.close()

def test_actual_recorder_exports_geometry_and_readable_delivery(recorder):
    ref,b=recorder;assert b['native']['native_geometry_sha256']==model_geometry_sha(ref.model_data()[0])
    assert b['native']['native_geometry_inventory']['geom_count']>100

def test_actual_step_guard_rejects_before_any_action(recorder):
    ref,b=recorder;m,d=ref.model_data();original=m.geom_pos[0].copy();ref.recording=dict(task_video_identity=b)
    try:
        m.geom_pos[0,0]+=.02
        with pytest.raises(VisibleBindingHold,match='binding changed'):ref.step(np.zeros(ref.env.action_dim))
    finally:m.geom_pos[0]=original;ref.recording=None

def test_actual_renderer_other_native_instance_rejected(recorder):
    ref,_=recorder;m,_=ref.model_data()
    with pytest.raises(VisibleBindingHold,match='another native model instance'):validate_render_model(copy.deepcopy(m),ref.renderer)

def test_actual_zero_action_frame_binds_model_state_camera_and_rgb(recorder):
    import hashlib
    ref,b=recorder;m,d=ref.model_data();ref.recording=dict(task_video_identity=b,n=0)
    try:
        image=ref.frame(b['camera']);captured=ref.recording['native_frame_binding']
        assert captured['native_model_geometry_sha256']==b['native']['native_geometry_sha256']
        assert captured['actual_qpos_sha256']==hashlib.sha256(d.qpos.tobytes()).hexdigest()
        assert captured['raw_rgb_sha256']==hashlib.sha256(image.tobytes()).hexdigest()
        assert b['schema']=='native-task-video-identity-v2-visible-geometry'
    finally:ref.recording=None
