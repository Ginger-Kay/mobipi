"""Replay development actions including recorded dock controller-reset events."""
import argparse
import json
from pathlib import Path
from teleop_reference import Reference, write_json, stamp
import numpy as np
import mujoco


def capture(ref, step, action):
    m,d=ref.model_data();arm=ref.robot.part_controllers['right']
    qids=np.asarray(arm.qpos_index,int);dofs=np.asarray(arm.qvel_index,int)
    joints=np.array([np.flatnonzero(m.jnt_qposadr==i)[0] for i in qids])
    actuators=np.asarray(ref.robot._ref_actuators_indexes_dict['right'],int)
    contacts=[]
    for i,c in enumerate(d.contact):
        names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g)) or '' for g in (c.geom1,c.geom2)]
        if not any(n.startswith(('robot0_','gripper0_','mobilebase0_')) for n in names):continue
        force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
        contacts.append(dict(names=names,distance=float(c.dist),contact_frame_wrench=force.tolist()))
    fixture=[]
    for j in range(m.njnt):
        name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) or ''
        if not name.endswith(('_microjoint','_slidejoint')):continue
        k=m.jnt_dofadr[j]
        fixture.append(dict(name=name,qpos=float(d.qpos[m.jnt_qposadr[j]]),qvel=float(d.qvel[k]),frictionloss=float(m.dof_frictionloss[k]),constraint_force=float(d.qfrc_constraint[k]),passive_force=float(d.qfrc_passive[k]),bias_force=float(d.qfrc_bias[k])))
    return dict(step=step,action=np.asarray(action).tolist(),arm_qpos=d.qpos[qids].tolist(),arm_qvel=d.qvel[dofs].tolist(),joint_margin=np.minimum(d.qpos[qids]-m.jnt_range[joints,0],m.jnt_range[joints,1]-d.qpos[qids]).tolist(),requested_torque=np.asarray(arm.torques).tolist(),applied_ctrl=d.ctrl[actuators].tolist(),ctrl_range=m.actuator_ctrlrange[actuators].tolist(),constraint_torque=d.qfrc_constraint[dofs].tolist(),nullspace_goal=np.asarray(arm.initial_joint).tolist(),controller_goal_world=(arm.origin_pos+arm.origin_ori@arm.goal_pos).tolist(),eef_pos=d.site_xpos[ref.robot.eef_site_id['right']].tolist(),jacobian_singular_values=np.linalg.svd(arm.J_full,compute_uv=False).tolist(),contacts=contacts,fixture=fixture)


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    attempt=Path(args.attempt).resolve()
    result=json.loads((attempt/'result.json').read_text())
    events=result['events']
    if any(e['event']!='dock_settled_reobserve_feedback_reset' for e in events):
        raise ValueError('Unknown control event; do not silently omit it')
    resets={e['step'] for e in events}
    reset_payloads={e['step']:e for e in events}
    ref=Reference(argparse.Namespace(output=args.output,task='CloseDrawer',layout=0,style=0,seed=7,self_test=True,source=None,replay_attempt=str(attempt),resume_attempt=None,width=1920,height=1080))
    original_step=ref.env.step
    count=0;applied=[];diagnostics=[]
    def step(action):
        nonlocal count
        if count in resets:
            ref.env._get_observations(force_update=True)
            ref.robot.composite_controller.update_state()
            ref.robot.part_controllers['right'].set_goal_update_mode('achieved')
            ref.robot.part_controllers['right'].set_goal(np.zeros(6))
            if 'arm_nullspace_goal' in reset_payloads[count]:
                goal=np.asarray(reset_payloads[count]['arm_nullspace_goal'],float)
                if goal.shape!=(7,) or not np.isfinite(goal).all():raise ValueError('Invalid nullspace event')
                ref.robot.part_controllers['right'].initial_joint=goal.copy()
            applied.append(count)
        value=original_step(action)
        if count%100==0 or count>=result['steps']-60:diagnostics.append(capture(ref,count,action))
        count+=1
        return value
    ref.env.step=step
    try:
        errors=ref.replay()
        write_json(Path(args.output)/'control-diagnostics.json',dict(scope='read-only snapshots during exact protocol action replay',samples=diagnostics))
        if set(applied)!=resets:raise AssertionError('Not all recorded events applied')
        write_json(Path(args.output)/'completed.json',dict(ended_at=stamp(),attempt=str(attempt),steps=count,applied_reset_steps=applied,max_state_abs_error=max(errors),checker_success=bool(ref.env._check_success()),scope='protocol-aware action replay; no trajectory correction or state injection'))
    finally:
        ref.env.step=original_step
        if ref.observation_renderer is not None:ref.observation_renderer.close()
        if ref.renderer is not None:ref.renderer.close()
        ref.env.close()

if __name__=='__main__':main()
