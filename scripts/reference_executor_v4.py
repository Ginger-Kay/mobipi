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
                     opening=float(state['target']['door']), grasp=float(grasp_states[i]))
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
        action[:6] = np.clip(local / np.asarray(arm.output_max), -.10, .10)
    action[6] = point['grasp'] if arm_enabled else -1.0
    action[11] = -1.0
    error = base_target-d.qpos[base.qpos_index]
    velocity = np.clip(error * 1.5, -.09, .09)
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


def run_route(ref, route, points, horizon):
    ref.route=route
    ref.begin()
    path=ref.recording['path']
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
    offset[:2] *= max(0.,1.-.15/max(np.linalg.norm(offset[:2]),1e-6))
    stow_local=d.xmat[body].reshape(3,3).T@offset
    stow_rot_local=d.xmat[body].reshape(3,3).T@d.site_xmat[site].reshape(3,3)
    index=0; stuck=0; settled=0; reason='horizon'; base_max_drift=0.
    dock=.5*(points[0]['base']+points[-1]['base'])
    initial_opening=float(ref.trace()['target']['door'])
    log=(path/'feedback.jsonl').open('w')
    try:
        for step in range(horizon):
            point=points[index]
            if stage in ('stow','navigate'):
                R=d.xmat[body].reshape(3,3)
                point=dict(pos=d.xpos[body]+R@stow_local,rot=R@stow_rot_local,grasp=-1.)
            target=dock if route=='D' else (initial if route=='E' else point['base'])
            if stage=='stow':target=initial
            action,pe,re,be=mapped_action(ref,point,target,True)
            if ref.env._check_success():
                action[:6]=0.;action[7:10]=0.
            if stage=='stow':
                stuck+=1
                if pe<.015 and re<.09:stage='navigate';stuck=0
            if stage=='navigate':
                settled=settled+1 if be<.008 and np.linalg.norm(d.qvel[base.qvel_index])<.005 else 0
                if settled>=20:
                    stage='manipulate'
                    # Stateless feedback has no observation history to carry across docking.
                    ref.env._get_observations(force_update=True)
                    ref.robot.composite_controller.update_state()
                    ref.robot.part_controllers['right'].set_goal_update_mode('achieved')
                    ref.robot.part_controllers['right'].set_goal(np.zeros(6))
                    ref.recording['events'].append(dict(step=step,event='dock_settled_reobserve_feedback_reset'))
                    index=next((i for i,p in enumerate(points) if p['opening'] < initial_opening-.01),0)
                    stuck=0
            elif stage=='manipulate':
                live=ref.trace()
                target_contact=any('finger' in str(c) and any(n in str(c) for n in ('microwave_main_group','stack_4_main_group_2')) for c in live['contacts'])
                articulation_reached=(point['opening'] < initial_opening-.02 and
                    float(live['target']['door']) <= point['opening']+.005 and target_contact)
                at_goal=(pe<.009 or articulation_reached) and re<.09 and (route!='A' or be<.012)
                if at_goal and index<len(points)-1:
                    index+=1;stuck=0
                else:stuck+=1
                if route in ('E','D'):
                    base_max_drift=max(base_max_drift,float(np.linalg.norm(d.qpos[base.qpos_index]-target)))
            log.write(json.dumps(dict(step=step,stage=stage,waypoint=index,pos_error=pe,rot_error=re,base_error=be))+'\n')
            ref.step(action)
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
                if stage=='manipulate' and 'finger' in robot[0] and world[0].startswith(('microwave_main_group','stack_4_main_group_2')):continue
                unsafe.append(contact)
            if unsafe:
                write_json(path/'contact-stop.json',dict(step=step,contacts=unsafe))
                reason='nonfinger_environment_contact';break
            # Treat stalled geometry as a valid development outcome, not mechanical retry.
            if stuck>=180:
                reason='tracking_stall';break
        ref.finish(reason)
    except BaseException:
        ref.finish('executor_exception');raise
    finally:log.close()
    write_json(path/'executor-result.json',dict(executor='reference-feedback-v4',route=route,reason=reason,
        base_max_drift_generalized=base_max_drift,waypoints_reached=index,total_waypoints=len(points),
        execution_scope='development',formal_train_ready=False,strict_semantics_verified=False,
        note='No collision safety certification; feedback path is reference-conditioned; no OBC features exported yet.'))
    return str(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    p.add_argument('--routes',default='A,E,D');p.add_argument('--horizon',type=int,default=1400)
    args=p.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    attempt=Path(args.reference).resolve();source=attempt.parent.parent
    # Isolated copy: recorder output must never be written inside human sources.
    dest=out/source.name;dest.mkdir()
    for name in ['model.xml','integration.npy','ep_meta.json','rng.json','source.json','target-binding.json']:
        shutil.copy2(source/name,dest/name)
    shutil.copy2(source.parent/'env_config.json',out/'env_config.json')
    write_json(out/'executor-spec.json',dict(created_at=stamp(),version='reference-feedback-v4',reference=str(attempt),
        source=str(source),routes=args.routes,horizon=args.horizon,source_sha256=hashlib.sha256((source/'integration.npy').read_bytes()).hexdigest(),
        scope='development_only',policy_replacement=True,feedback='achieved OSC pose deltas; measured waypoint advancement; generalized-base position servo',
        limitations=['same-source prototype','no swept collision validator','no formal OBC feature export','D uses retract then relative-pose hold while navigating to reference midpoint; dock/path not continuously validated'],
        inputs='live simulator robot kinematics, current target joint and contact plus frozen demonstration geometry; no new-route future outcome input'))
    cfg=json.loads((out/'env_config.json').read_text())
    ref=Reference(argparse.Namespace(output=str(out),task=cfg['env_name'],layout=0,style=0,seed=7,self_test=True,
        source=str(dest),replay_attempt=None,resume_attempt=None,width=1920,height=1080))
    ref.label='autonomous_development_reference_feedback_v1'
    try:
        ref.restore();points=compile_path(ref,attempt);write_json(out/'waypoints.json',points)
        results=[]
        for route in args.routes.split(','):
            if route not in ('A','E','D'):raise ValueError(route)
            results.append(run_route(ref,route,points,args.horizon))
        write_json(out/'completed.json',dict(ended_at=stamp(),attempts=results))
    except BaseException:
        write_json(out/'error.json',dict(time=stamp(),traceback=traceback.format_exc()));raise
    finally:
        if ref.observation_renderer is not None:ref.observation_renderer.close()
        if ref.renderer is not None:ref.renderer.close()
        ref.env.close()

if __name__=='__main__':main()
