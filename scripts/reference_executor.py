"""Development-only feedback executor learned geometrically from a human reference.

No recorded action is replayed. EEF/base waypoints are advanced using measured
tracking error; E/D/A share the same arm path. Not the frozen-policy executor,
not an OBC training collector, and not validated on unseen sources.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import traceback
import numpy as np
import mujoco
import h5py
from scipy.spatial.transform import Rotation
from teleop_reference import Reference, write_json, stamp
from reference_control_diagnostics import capture
from reference_planning import compile_candidates
from mobiwam.reference_dispatch import rejection_reason, target_finger_contact
from reference_geometry import plan_dock, PalmClearance, adjust_reference, translation_limit
from mobiwam.reference_prefix_safety import JointMarginMonitor, JointMarginStop, GuardedIntegration
from mobiwam.reference_formal_substep import FormalSubstepMonitor, FormalSafetyStop


class ProgressWatch:
    """Count steps without a meaningful improvement, not total phase duration."""
    def __init__(self, tolerance=.001):
        self.tolerance=tolerance;self.key=None;self.best=float('inf');self.idle=0
    def update(self,key,error):
        if key!=self.key or error<self.best-self.tolerance:
            self.key=key;self.best=error;self.idle=0
        else:self.idle+=1
        return self.idle


def orientation_error(target, current):
    return Rotation.from_matrix(target @ current.T).as_rotvec()


def compile_path(ref, attempt):
    records = [json.loads(s) for s in (attempt / 'trace.jsonl').read_text().splitlines()]
    m, _ = ref.model_data()
    scratch = mujoco.MjData(m)
    site = ref.robot.eef_site_id['right']
    base = ref.robot.part_controllers['base']
    with h5py.File(attempt / 'demo.hdf5') as h:
        grasp_states=h['data/demo_0/actions'][:,6]
    binding=json.loads((ref.root/'target-binding.json').read_text())
    target_joint=next(j for j in binding['joints'] if j['name'].endswith(('_microjoint','_slidejoint')))
    address=target_joint['qpos_address']
    first_success=next((i for i,r in enumerate(records) if r['after']['success']),len(records)-1)
    records=records[:first_success+1]
    points = []
    # Reference is a development input; its outcomes never become new-route labels.
    for i, record in enumerate(records):
        if i % 6 and i != len(records)-1 and (i==0 or grasp_states[i]==grasp_states[i-1]):
            continue
        state = record['after']
        scratch.qpos[:] = state['qpos']
        mujoco.mj_forward(m, scratch)
        point = dict(reference_step=i, pos=scratch.site_xpos[site].copy(),
                     rot=scratch.site_xmat[site].reshape(3, 3).copy(),
                     base=scratch.qpos[base.qpos_index].copy(),
                     opening=float(state['target']['door']), grasp=float(grasp_states[i]),
                     arm_qpos=np.asarray(state['arm_qpos']),fixture_qpos_address=address,fixture_qpos=state['qpos'][address])
        point=adjust_reference(ref,scratch,point)
        if points and point['grasp']==points[-1]['grasp'] and np.linalg.norm(point['pos']-points[-1]['pos']) < .006 and np.linalg.norm(point['base']-points[-1]['base']) < .006 and np.linalg.norm(orientation_error(point['rot'],points[-1]['rot'])) < .03:
            continue
        points.append(point)
    if not points:
        raise ValueError('Empty reference path')
    return points


def mapped_action(ref, point, base_target, arm_enabled=True):
    """World EEF pose feedback and generalized-base servo, achieved OSC deltas."""
    m, d = ref.model_data()
    arm = ref.robot.part_controllers['right']
    base = ref.robot.part_controllers['base']
    site = ref.robot.eef_site_id['right']
    pos_error = point['pos'] - d.site_xpos[site]
    rot_error = orientation_error(point['rot'], d.site_xmat[site].reshape(3,3))
    action = np.zeros(ref.env.action_dim)
    if arm_enabled:
        local = np.r_[arm.origin_ori.T @ pos_error, arm.origin_ori.T @ rot_error]
        limit=translation_limit(ref)
        action[:3] = np.clip(local[:3] / np.asarray(arm.output_max)[:3], -limit, limit)
        action[3:6] = np.clip(local[3:6] / np.asarray(arm.output_max)[3:6], -.10, .10)
    action[6] = point['grasp'] if arm_enabled else -1.0
    action[11] = -1.0
    error = base_target-d.qpos[base.qpos_index]
    base_speed=.015 if ref.args.task=='CloseDrawer' else .09
    velocity = np.clip(error * 1.5, -base_speed, base_speed)
    velocity[np.abs(error)<.003] = 0
    ids = np.asarray(ref.robot._ref_actuators_indexes_dict['base'],int)
    friction = m.dof_frictionloss[base.qvel_index] / m.actuator_gainprm[ids,0]
    requested = velocity + np.where(np.abs(velocity)>1e-6,np.sign(velocity)*friction,0)
    goal = requested / (.5*(base.actuator_max-base.actuator_min))
    _, ori = base.get_base_pose()
    theta = np.arctan2(ori[1,0],ori[0,0])-np.arctan2(base.init_ori[1,0],base.init_ori[0,0])
    mapping = np.array([[-np.sin(theta),np.cos(theta),0],[np.cos(theta),np.sin(theta),0],[0,0,1]])
    action[7:10] = np.linalg.solve(mapping,goal)
    return action, float(np.linalg.norm(pos_error)), float(np.linalg.norm(rot_error)), float(np.linalg.norm(error))


def run_route(ref, route, points, horizon, *, execution_scope="development"):
    if execution_scope not in ("development", "DR-v0.4_formal_candidate_pending_audit"):
        raise ValueError("unrecognized execution scope")
    fixture=ref.env.drawer if ref.args.task=='CloseDrawer' else ref.env.door_fxtr
    target_name=fixture.name
    binding=json.loads((ref.source/'target-binding.json').read_text())
    if binding['fixture_name']!=target_name:
        raise ValueError('Source target binding differs from live fixture')
    ref.route=route
    ref.base_locked=route in ('E','D')
    ref.begin()
    path=ref.recording['path']
    if ref.args.task=='CloseDrawer' and target_name=='stack_4_main_group_2':
        camera=dict(lookat=[3.8,-1.0,.65],distance=2.4,azimuth=90.,elevation=-15.)
        ref.apply_camera(camera);ref.recording['camera']=camera
    clearance= PalmClearance(ref)
    m,d=ref.model_data()
    base=ref.robot.part_controllers['base']
    initial=d.qpos[base.qpos_index].copy()
    baseline=ref.integration().copy()
    expected=np.load(ref.source/'integration.npy')
    write_json(path/'restore-receipt.json',dict(max_integration_abs_error=float(np.max(np.abs(baseline-expected))),source=str(ref.source),time=stamp()))
    stage='stow' if route=='D' else 'manipulate'
    body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,ref.base_body)
    site=ref.robot.eef_site_id['right']
    offset=d.site_xpos[site]-d.xpos[body]
    offset[:2] *= max(0.,1.-.25/max(np.linalg.norm(offset[:2]),1e-6))
    stow_local=d.xmat[body].reshape(3,3).T@offset
    stow_rot_local=d.xmat[body].reshape(3,3).T@d.site_xmat[site].reshape(3,3)
    if route=='D' and 'stow_target' in ref.dock_plan['selected']:
        from reference_stow import load_stow
        stow_local,stow_rot_local,stow_q=load_stow(ref,ref.dock_plan['selected']['stow_target'])
        from mobiwam.reference_controller_events import emit_event
        emit_event(ref,dict(event='initial_stow_nullspace',step=0,timing='before_action',arm_nullspace_goal=stow_q.tolist()))
        ref.recording['initial_stow_required']=True
    watch=ProgressWatch()
    grip_waypoint=None;grip_count=0
    index=0; stuck=0; settled=0; reason='horizon'; base_max_drift=0.
    dock=np.asarray(ref.dock_plan['selected']['dock'])
    margin_guard=None
    formal_scope=execution_scope=='DR-v0.4_formal_candidate_pending_audit'
    formal_guard=FormalSubstepMonitor(ref,target_name) if formal_scope else None
    if route=='D' or formal_scope:
        arm=ref.robot.part_controllers['right']
        joints=[int(np.flatnonzero(m.jnt_qposadr==i)[0]) for i in arm.qpos_index]
        margin_guard=JointMarginMonitor(arm.qpos_index,m.jnt_range[joints],
            [mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) for j in joints])
    initial_opening=float(ref.trace()['target']['door'])
    log=(path/'feedback.jsonl').open('w')
    control_log=(path/'control.jsonl').open('w')
    try:
        if margin_guard is not None:
            failure=margin_guard.observe(d.qpos,step=0,substep=None,phase=stage,when='initial')
            if failure:raise JointMarginStop(failure)
        for step in range(horizon):
            point=points[index]
            if stage in ('stow','navigate'):
                R=d.xmat[body].reshape(3,3)
                point=dict(pos=d.xpos[body]+R@stow_local,rot=R@stow_rot_local,grasp=-1.)
            target=dock if route=='D' else (initial if route=='E' else point['base'])
            if stage=='stow':target=initial
            action,pe,re,be=mapped_action(ref,point,target,True)
            action,clearance_receipt=clearance.apply(action)
            progress_idle=watch.update((stage,index),pe+.1*re+be)
            if ref.env._check_success():
                action[:6]=0.;action[7:10]=0.
            if stage=='stow':
                stuck+=1
                if pe<.01 and re<.09:stage='navigate';stuck=0
            if stage=='navigate':
                settled=settled+1 if be<.008 and np.linalg.norm(d.qvel[base.qvel_index])<.005 else 0
                if settled>=20:
                    stage='manipulate'
                    # Stateless feedback has no observation history to carry across docking.
                    from mobiwam.reference_controller_events import emit_event
                    nullspace=np.asarray(ref.dock_plan['selected']['arm_nullspace_goal'])
                    emit_event(ref,dict(step=step,timing='before_action',event='dock_settled_reobserve_feedback_reset',arm_nullspace_goal=nullspace.tolist()))
                    # Reacquire with the shared open-gripper approach; never jump to an already-closing pose.
                    index=0
                    stuck=0
            elif stage=='manipulate':
                live=ref.trace()
                target_contact=any(target_finger_contact(c,target_name) for c in live['contacts'])
                articulation_reached=(index>0 and point['grasp']>0 and points[index-1]['grasp']>0 and
                    float(live['target']['door']) <= point['opening']+.005 and target_contact)
                handle_fingers=set()
                for contact in live['contacts']:
                    names=[contact.get('geom1') or '',contact.get('geom2') or '']
                    if not target_finger_contact(contact,target_name,handle_only=True):continue
                    for finger in (1,2):
                        if any('gripper0_right_finger'+str(finger) in n for n in names):handle_fingers.add(finger)
                is_grasp_waypoint=index>0 and point['grasp']>0 and points[index-1]['grasp']<0
                if grip_waypoint!=index:grip_waypoint=index;grip_count=0
                grip_count=grip_count+1 if is_grasp_waypoint and len(handle_fingers)==2 else 0
                grasp_reached=grip_count>=3
                at_goal=(pe<.009 or articulation_reached or grasp_reached) and re<.09 and (route!='A' or be<.012)
                if at_goal and index<len(points)-1:
                    index+=1;stuck=0
                else:stuck+=1
                if route in ('E','D'):
                    base_max_drift=max(base_max_drift,float(np.linalg.norm(d.qpos[base.qpos_index]-target)))
            log.write(json.dumps(dict(step=step,stage=stage,waypoint=index,pos_error=pe,rot_error=re,base_error=be,progress_idle=progress_idle,palm_clearance=clearance_receipt))+'\n')
            if margin_guard is None:
                ref.step(action)
            else:
                before_step=ref.integration().copy()
                integration_guard=GuardedIntegration(ref.env.sim,d,margin_guard,
                    lite_physics=ref.env.lite_physics,step=step,phase=stage)
                if formal_guard is not None:formal_guard.set_boundary(step,stage)
                try:
                    if formal_guard is not None:
                        with formal_guard:
                            with integration_guard:ref.step(action)
                    else:
                        with integration_guard:ref.step(action)
                except (JointMarginStop,FormalSafetyStop):
                    np.savez_compressed(path/'partial-control-step.npz',
                        initial_integration=before_step,terminal_integration=ref.integration(),
                        attempted_action=action,completed_physics_substeps=(len(formal_guard.phases)
                          if formal_guard is not None else integration_guard.completed_substeps))
                    raise
            if step%20==0:control_log.write(json.dumps(capture(ref,step,action))+'\n')
            if step%100==0:
                log.flush();print(route,step,stage,index,'errors',round(pe,4),round(be,4),flush=True)
            if not ref.recording:
                reason='checker_success_10_steps';break
            # Stop on observed non-finger environmental contact. This is a
            # post-step development guard, not continuous collision certification.
            contacts=ref.trace()['contacts']
            unsafe=[]
            for contact in contacts:
                names=[contact['geom1'] or '',contact['geom2'] or '']
                robot=[n for n in names if n.startswith(('robot0_','gripper0_','mobilebase0_'))]
                world=[n for n in names if n not in robot]
                if len(robot)!=1 or not world or 'floor' in world[0]:continue
                if stage=='manipulate' and target_finger_contact(contact,target_name):continue
                unsafe.append(contact)
            if unsafe:
                write_json(path/'contact-stop.json',dict(step=step,contacts=unsafe))
                reason='pre_dock_environment_contact' if stage!='manipulate' else 'nonfinger_environment_contact';break
            # Treat stalled geometry as a valid development outcome, not mechanical retry.
            if progress_idle>=180:
                reason='tracking_stall';break
        ref.finish(reason)
    except FormalSafetyStop as exc:
        reason='native_forbidden_contact_stop'
        write_json(path/'formal-substep-stop.json',exc.failure)
        ref.finish(reason)
    except JointMarginStop as exc:
        reason='joint_margin_stop'
        write_json(path/'joint-margin-stop.json',dict(failure=exc.failure,monitor=margin_guard.receipt(),
            partial_step_file='partial-control-step.npz' if (path/'partial-control-step.npz').exists() else None,
            replay_scope='HDF5 contains completed control steps only; partial step is separate and not a standard replay sample'))
        ref.finish(reason)
    except BaseException:
        ref.finish('executor_exception');raise
    finally:
        log.close();control_log.close()
        if margin_guard is not None:write_json(path/'joint-margin-monitor.json',margin_guard.receipt())
        if formal_guard is not None:write_json(path/'formal-native-substeps-receipt.json',formal_guard.save(path))
    write_json(path/'executor-result.json',dict(executor='reference-feedback-v17',route=route,reason=reason,
        base_max_drift_generalized=base_max_drift,waypoints_reached=index,total_waypoints=len(points),
        execution_scope=execution_scope,formal_train_ready=False,strict_semantics_verified=False,
        note='No collision safety certification; feedback path is reference-conditioned; all outcomes need original video and substep audit before eligibility.'))
    return str(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    p.add_argument('--source');p.add_argument('--plan-only',action='store_true')
    p.add_argument('--require-full-plan',action='store_true',help='Reject any requested route without a hard-valid complete geometric plan')
    p.add_argument('--routes',default='A,E,D');p.add_argument('--horizon',type=int,default=2400)
    args=p.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    attempt=Path(args.reference).resolve();original_source=attempt.parent.parent
    source=Path(args.source).resolve() if args.source else original_source
    if (source/'model.xml').read_bytes()!=(original_source/'model.xml').read_bytes():raise ValueError('Source model changed')
    if json.loads((source.parent/'env_config.json').read_text())!=json.loads((original_source.parent/'env_config.json').read_text()):raise ValueError('Source environment changed')
    # Isolated copy: recorder output must never be written inside human sources.
    dest=out/source.name;dest.mkdir()
    for name in ['model.xml','integration.npy','ep_meta.json','rng.json','source.json','target-binding.json']:
        shutil.copy2(source/name,dest/name)
    shutil.copy2(source.parent/'env_config.json',out/'env_config.json')
    write_json(out/'executor-spec.json',dict(created_at=stamp(),version='reference-feedback-v17',reference=str(attempt),
        source=str(source),routes=args.routes,horizon=args.horizon,require_full_plan=args.require_full_plan,source_sha256=hashlib.sha256((source/'integration.npy').read_bytes()).hexdigest(),
        scope='development_only',policy_replacement=True,base_speed_caps=dict(CloseDrawer=.015,CloseSingleDoor=.09),translation_action_caps=dict(CloseDrawer=.10,CloseSingleDoor=.20,CloseSingleDoor_locked_base_bilateral_handle_contact=.40),contact_model='slide-joint co-motion only under bilateral pad contact; other fixtures retain static guard',feedback='achieved OSC pose deltas; measured waypoint advancement; generalized-base position servo',
        limitations=['same-source prototype','palm local linear guard, not full swept collision validator','development 21D planning feature export; formal binding pending','D checks Source-to-stow-to-dock geometric swept path; OSC execution can deviate'],
        inputs='live simulator robot kinematics, current target joint and contact plus frozen demonstration geometry; no new-route future outcome input'))
    cfg=json.loads((out/'env_config.json').read_text())
    ref=Reference(argparse.Namespace(output=str(out),task=cfg['env_name'],layout=0,style=0,seed=7,self_test=True,
        source=str(dest),replay_attempt=None,resume_attempt=None,width=1920,height=1080))
    ref.label='autonomous_development_reference_feedback_v17'
    try:
        ref.restore();points=compile_path(ref,attempt);write_json(out/'waypoints.json',points)
        ref.dock_plan=plan_dock(ref,points)
        preflight=compile_candidates(ref,points,ref.dock_plan,out/'planning')
        write_json(out/'dock-plan.json',ref.dock_plan)
        write_json(out/'recording-provenance.json',dict(data_kind='autonomous_development',version='reference-feedback-v17',formal_train_ready=False))
        results=[]
        for route in ([] if args.plan_only else args.routes.split(',')):
            if route not in ('A','E','D'):raise ValueError(route)
            rejection=rejection_reason(preflight,route,args.require_full_plan)
            if rejection:
                write_json(out/f'{route}-planning-rejected.json',dict(reason=rejection,new_route_outcome=False));continue
            results.append(run_route(ref,route,points,args.horizon))
        write_json(out/'completed.json',dict(ended_at=stamp(),attempts=results))
    except BaseException:
        write_json(out/'error.json',dict(time=stamp(),traceback=traceback.format_exc()));raise
    finally:
        if ref.observation_renderer is not None:ref.observation_renderer.close()
        if ref.renderer is not None:ref.renderer.close()
        ref.env.close()

if __name__=='__main__':main()
