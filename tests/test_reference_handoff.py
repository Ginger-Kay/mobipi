import hashlib
import json
import numpy as np
import pytest
from mobiwam.reference_handoff import load_preview_prefix
from mobiwam.reference_prefix_safety import GuardedIntegration, JointMarginMonitor, JointMarginStop


def saved(tmp_path):
    states=np.array([[.5],[.4],[.3]])
    archive=tmp_path/'prefix-prediction.npz'
    np.savez(archive,qpos=states,actions=np.zeros((2,11)))
    trace=tmp_path/'trace.json';trace.write_text(json.dumps([{'phase':'stow'},{'phase':'navigate'}]))
    result=dict(valid=True,candidate_id=6,control_steps=2,failure=None,
        joint_margin=dict(required_strictly_greater_than_rad=.015,minimum=dict(margin_rad=.3)),
        prefix_qpos_actions_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        prefix_trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
    (tmp_path/'result.json').write_text(json.dumps(result))
    return states,result


def test_actual_terminal_is_preserved(tmp_path):
    states,_=saved(tmp_path)
    actual,phases,errors=load_preview_prefix(tmp_path,states[0],6)
    np.testing.assert_array_equal(actual,states)
    assert phases==['stow','navigate']
    np.testing.assert_array_equal(errors,np.zeros((2,2)))


@pytest.mark.parametrize('kind',['source','id','archive','trace','margin','nonfinite','missing_seal','invalid'])
def test_mismatched_preview_is_rejected(tmp_path,kind):
    states,result=saved(tmp_path);initial=states[0];candidate=6
    if kind=='source':initial=np.array([.6])
    if kind=='id':candidate=5
    if kind=='archive':(tmp_path/'prefix-prediction.npz').write_bytes(b'changed')
    if kind=='trace':(tmp_path/'trace.json').write_text('[]')
    if kind=='margin':result['joint_margin']['minimum']['margin_rad']=.015
    if kind=='nonfinite':result['joint_margin']['minimum']['margin_rad']=float('nan')
    if kind=='missing_seal':del result['prefix_qpos_actions_sha256']
    if kind=='invalid':result['valid']=False
    (tmp_path/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError):load_preview_prefix(tmp_path,initial,candidate)


@pytest.mark.parametrize('lite',[False,True])
@pytest.mark.parametrize('initial,delta,expected',[ (.01,0.,0),(.5,-.49,1),(.5,-.01,3)])
def test_integration_guard_stops_and_restores(lite,initial,delta,expected):
    from types import SimpleNamespace as NS
    data=NS(qpos=np.array([initial]));calls=[]
    class Sim:
        def step(self):
            calls.append('step');data.qpos[0]+=delta
        def step2(self):
            calls.append('step2');data.qpos[0]+=delta
    sim=Sim();name='step2' if lite else 'step';original=getattr(sim,name)
    monitor=JointMarginMonitor([0],[[0.,1.]],['joint'])
    guard=GuardedIntegration(sim,data,monitor,lite_physics=lite,step=3,phase='manipulate')
    def run():
        with guard:
            for _ in range(3):getattr(sim,name)()
    if expected<3:
        with pytest.raises(JointMarginStop):run()
    else:run()
    assert len(calls)==guard.completed_substeps==expected
    assert getattr(sim,name)==original and name not in vars(sim)


def test_guard_restores_instance_override_on_exception():
    from types import SimpleNamespace as NS
    def error():raise RuntimeError('controller error')
    sim=NS(step2=error);data=NS(qpos=np.array([.5]))
    monitor=JointMarginMonitor([0],[[0.,1.]],['joint'])
    with pytest.raises(RuntimeError,match='controller error'):
        with GuardedIntegration(sim,data,monitor,lite_physics=True,step=0,phase='stow'):sim.step2()
    assert sim.step2 is error


@pytest.mark.parametrize('lite',[False,True])
def test_native_mujoco_integration_unchanged_when_safe(lite):
    import mujoco
    from types import SimpleNamespace as NS
    model=mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
    <option gravity="0 0 0"/><worldbody><body><joint range="0 1"/>
    <geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>''')
    data=mujoco.MjData(model);baseline=mujoco.MjData(model)
    for state in (data,baseline):state.qpos[0]=.5;state.qvel[0]=.1
    class Sim:
        def step(self):mujoco.mj_step(model,data)
        def step2(self):mujoco.mj_step2(model,data)
    sim=Sim();monitor=JointMarginMonitor([0],[[0.,1.]],['joint'])
    with GuardedIntegration(sim,data,monitor,lite_physics=lite,step=0,phase='navigate') as guard:
        for _ in range(3):
            if lite:
                mujoco.mj_step1(model,data);sim.step2()
                mujoco.mj_step1(model,baseline);mujoco.mj_step2(model,baseline)
            else:
                sim.step();mujoco.mj_step(model,baseline)
    np.testing.assert_array_equal(data.qpos,baseline.qpos)
    np.testing.assert_array_equal(data.qvel,baseline.qvel)
    assert data.time==baseline.time and guard.completed_substeps==3


def test_preview_and_execution_reset_have_identical_controller_effects():
    import ast
    from pathlib import Path
    from types import SimpleNamespace as NS
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    target=np.arange(7,dtype=float)/10
    effects=[]
    for filename in ('reference_prefix_preview.py','reference_executor.py'):
        calls=[]
        arm=NS(set_goal_update_mode=lambda mode:calls.append(('mode',mode)),
               set_goal=lambda goal:calls.append(('goal',np.asarray(goal).tolist())))
        env=NS(_get_observations=lambda **kw:calls.append(('observe',kw)))
        robot=NS(composite_controller=NS(update_state=lambda:calls.append(('update',))),
                 part_controllers={'right':arm})
        candidate=dict(arm_nullspace_goal=target.tolist())
        ref=NS(env=env,robot=robot,dock_plan={'selected':candidate},recording={'events':[]})
        tree=ast.parse((scripts/filename).read_text())
        blocks=[node for node in ast.walk(tree) if isinstance(node,ast.If)
            and isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name)
            and node.test.left.id=='settled']
        assert len(blocks)==1
        namespace=dict(np=np,env=env,preview=ref,ref=ref,candidate=candidate,step=216)
        exec(compile(ast.Module(body=blocks[0].body,type_ignores=[]),filename,'exec'),namespace)
        effects.append((calls,arm.initial_joint.tolist()))
    assert effects[0]==effects[1]
    assert effects[0][0]==[('observe',{'force_update':True}),('update',),('mode','achieved'),('goal',[0.]*6)]
    assert effects[0][1]==target.tolist()
