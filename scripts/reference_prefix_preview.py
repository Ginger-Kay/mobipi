"""Isolated model-based D prefix prediction, with no task outcome query.

Use the real low-level controller and the fork's native physics substep order.
Never call env.step/reward/_post_action/task checker during prediction. This is
a simulator planning probe (not zero physics steps), not a new task outcome.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import mujoco
from teleop_reference import Reference, stamp


def preview_prefix(ref,candidate,output,horizon=1200):
    from reference_executor import mapped_action,ProgressWatch
    from reference_geometry import PalmClearance
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    preview=Reference(argparse.Namespace(output=str(output),task=ref.args.task,layout=0,style=0,
        seed=7,self_test=True,source=str(ref.source),replay_attempt=None,resume_attempt=None,width=640,height=360))
    preview.base_locked=True;preview.route='D';preview.restore()
    # Source validation in Reference initialization may query its checker.
    # From here onward, querying a task outcome is an explicit error.
    def denied(*a,**kw):raise RuntimeError('task outcome forbidden during prefix prediction')
    preview.env._check_success=denied;preview.env.reward=denied
    m,d=preview.model_data();env=preview.env
    base=preview.robot.part_controllers['base'];site=preview.robot.eef_site_id['right']
    body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,preview.base_body)
    initial=d.qpos[base.qpos_index].copy();dock=np.asarray(candidate['dock'])
    offset=d.site_xpos[site]-d.xpos[body]
    offset[:2]*=max(0.,1.-.25/max(np.linalg.norm(offset[:2]),1e-6))
    local=d.xmat[body].reshape(3,3).T@offset
    localrot=d.xmat[body].reshape(3,3).T@d.site_xmat[site].reshape(3,3)
    guard=PalmClearance(preview);watch=ProgressWatch();phase='stow';settled=0
    states=[d.qpos.copy()];actions=[];trace=[];substeps=0;failure=None;ended=False
    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(m.ngeom)]
    def contacts(data):
        forbidden=[]
        for c in data.contact:
            a,b=names[c.geom1],names[c.geom2]
            ra=a.startswith(('robot0_','gripper0_','mobilebase0_'));rb=b.startswith(('robot0_','gripper0_','mobilebase0_'))
            if not (ra or rb):continue
            if ra and rb:
                forbidden.append(dict(pair=[a,b],distance_m=float(c.dist),kind='self_contact'));continue
            robot=a if ra else b;world=b if ra else a
            if 'floor' in world and robot.startswith('mobilebase0_'):continue
            forbidden.append(dict(pair=[a,b],distance_m=float(c.dist),kind='environment_contact'))
        return forbidden
    try:
        for step in range(horizon):
            R=d.xmat[body].reshape(3,3)
            point=dict(pos=d.xpos[body]+R@local,rot=R@localrot,grasp=-1.)
            target=initial if phase=='stow' else dock
            action,pe,re,be=mapped_action(preview,point,target,True);action,_=guard.apply(action)
            idle=watch.update((phase,0),pe+.1*re+be)
            if phase=='stow' and pe<.01 and re<.09:phase='navigate'
            finished=False
            if phase=='navigate':
                settled=settled+1 if be<.008 and np.linalg.norm(d.qvel[base.qvel_index])<.005 else 0
                if settled>=20:
                    env._get_observations(force_update=True)
                    preview.robot.composite_controller.update_state()
                    arm=preview.robot.part_controllers['right'];arm.set_goal_update_mode('achieved')
                    arm.set_goal(np.zeros(6));arm.initial_joint=np.asarray(candidate['arm_nullspace_goal']).copy()
                    finished=True
            # Same integration and observable-update order as MujocoEnv.step,
            # omitting _post_action, reward and task-success evaluation.
            env.timestep+=1;policy=True
            for substep in range(int(env.control_timestep/env.model_timestep)):
                if env.lite_physics:env.sim.step1()
                else:env.sim.forward()
                bad=contacts(d)
                if bad:
                    failure=dict(kind='predicted_contact',step=step,substep=substep,phase=phase,contacts=bad);break
                env._pre_action(action,policy)
                if env.lite_physics:env.sim.step2()
                else:env.sim.step()
                env._update_observables();policy=False;substeps+=1
            env.cur_time+=env.control_timestep
            env._get_observations()
            states.append(d.qpos.copy());actions.append(action.copy())
            trace.append(dict(step=step,phase=phase,pos_error_m=pe,rotation_error_rad=re,base_error=be))
            if failure:break
            if finished:
                # The last integration result has no following step1/forward.
                # Check its contact geometry on scratch data, preserving exact
                # controller/physics state and avoiding an unchecked endpoint.
                terminal=mujoco.MjData(m);terminal.qpos[:]=d.qpos
                terminal.mocap_pos[:]=d.mocap_pos;terminal.mocap_quat[:]=d.mocap_quat
                mujoco.mj_fwdPosition(m,terminal);bad=contacts(terminal)
                if bad:failure=dict(kind='predicted_terminal_contact',step=step,phase='settle',contacts=bad)
                else:ended=True
                break
            if idle>=180:
                failure=dict(kind='predicted_tracking_stall',step=step,phase=phase);break
        if not ended and failure is None:failure=dict(kind='prefix_prediction_horizon',horizon=horizon)
        result=dict(created_at=stamp(),valid=ended and failure is None,candidate_id=candidate['id'],
            control_steps=len(actions),physics_substeps=substeps,failure=failure,
            scope='independent native-physics planning preview through settled controller reset; no task outcome query',
            outcome_queries_during_prediction=0,environment_step_calls=0,initial_source_validation_may_query_checker=True,
            collision_sampling='every native physics substep plus terminal configuration on scratch data; includes self and forbidden pre-dock finger contacts',
            theoretical_continuous_dynamics_certificate=False)
        np.savez_compressed(output/'prefix-prediction.npz',qpos=np.asarray(states),actions=np.asarray(actions))
        (output/'trace.json').write_text(json.dumps(trace,indent=2))
        (output/'result.json').write_text(json.dumps(result,indent=2))
        return result
    finally:
        if preview.observation_renderer is not None:preview.observation_renderer.close()
        if preview.renderer is not None:preview.renderer.close()
        preview.env.close()
