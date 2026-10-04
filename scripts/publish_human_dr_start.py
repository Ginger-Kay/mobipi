"""Publish a derived drawer practice start and one CPU-only 20-step hold."""
import argparse
import os
from pathlib import Path
import mujoco
import numpy as np
from human_scene_pilot import PilotReference, make_args, load, write_json, stamp, code_commit, key_actions
from teleop_reference import Reference
from mobiwam.reference_formal_substep import FormalSubstepMonitor
from mobiwam.reference_prefix_safety import JointMarginMonitor, GuardedIntegration
from mobiwam.visible_object_binding import model_geometry_sha


def publish(batch, run):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only preparation'
    old = load(batch/'scenes/DR-PILOT/pilot-v1/pilot.json')
    design = load(run/'static-search-side/result.json')
    selected = design['selected']
    assert selected and selected['accepted']
    root = batch/'scenes/DR-PILOT/pilot-v2-precontact'
    root.mkdir(exist_ok=False)
    ref = PilotReference(make_args(root, old['task'], old['environment_seed'], Path(old['source'])), old.copy())
    try:
        ref.restore(); ref.bind()
        m, d = ref.model_data()
        before = d.qpos.copy(); geometry = model_geometry_sha(m)
        new = np.load(run/f'static-search-side/candidate-{selected["index"]:02d}-qpos.npy')
        assert np.array_equal(new[11:], before[11:])
        d.qpos[:] = new; d.qvel[:13] = 0; d.qacc_warmstart[:13] = 0
        mujoco.mj_forward(m, d); d.qacc_warmstart[:13] = 0
        assert model_geometry_sha(m) == geometry
        ref.robot.composite_controller.update_state()
        for controller in ref.robot.part_controllers.values():
            try: controller.update(force=True)
            except TypeError: controller.update()
        ref.robot.composite_controller.reset()
        arm = ref.robot.part_controllers['right']
        arm.set_goal_update_mode('achieved'); arm.set_goal(np.zeros(arm.control_dim))
        check = FormalSubstepMonitor(ref, ref.native['fixture_name'])
        assert not list(check.forbidden_contacts())
        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'gripper0_right_grip_site')
        finger = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'gripper0_right_finger_joint1')
        handle = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'stack_4_main_group_3_door_handle_handle')
        axis = d.xaxis[finger].copy(); longitudinal = d.geom_xmat[handle].reshape(3,3)[:,2]
        assert abs(axis[2]) > .999 and abs(np.dot(axis, longitudinal)) < .001
        assert np.linalg.norm(d.site_xpos[site]-d.geom_xpos[handle]) > .119
        ref.label = 'human_practice_start_variant_drawer_precontact_v2'
        ref.verify_target(); source = Reference.freeze(ref)
        ref.restore(); ref.bind()
        assert np.array_equal(ref.model_data()[1].qpos, new)
        cfg = {**old, 'config_version':'pilot-v2-precontact', 'source':str(source),
               'created_at':stamp(), 'code_commit':code_commit(),
               'target_binding':str(source/'target-binding.json'), 'primary_enabled':False, 'frozen_at':None,
               'source_lineage':dict(parent_source=old['source'], family=old['scene_family_id'], same_environment=True,
                   independent_source_increment=0, reason='User reports prolonged wrist adjustment; align actual finger closing axis across horizontal drawer handle',
                   robot_start_design=selected, precontact_distance_m=.12, fixture_and_nonrobot_qpos_unchanged=True,
                   open_gripper=True, finger_qpos_unchanged=True, initial_native_contact_free=True),
               'intended_use':'unfrozen human A practice; derived robot start, not independent environment or primary paired result',
               'restore_max_abs_error':ref.restore_receipt['max_abs_error'], 'restore_check':ref.restore_receipt, 'new_task_actions':0}
        write_json(root/'pilot.json',cfg); write_json(root/'native-identity.json',ref.native)
        write_json(root/'preparation-receipt.json',dict(at=stamp(), design=selected, geometry_unchanged=True,
            nonrobot_and_finger_qpos_unchanged=True, restore=ref.restore_receipt,
            world_finger_closing_axis=axis, world_handle_axis=longitudinal,
            origin='Static initial-condition design, not an executed approach', new_physics_steps=0))
        m,d = ref.model_data(); initial = d.qpos.copy(); eef = d.site_xpos[site].copy()
        arm = ref.robot.part_controllers['right']
        ids = [int(np.flatnonzero(m.jnt_qposadr==i)[0]) for i in arm.qpos_index]
        joint = JointMarginMonitor(arm.qpos_index,m.jnt_range[ids],[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in ids])
        guard = FormalSubstepMonitor(ref,ref.native['fixture_name'])
        action = ref.robot.create_action_vector(key_actions(set(),False,True))
        states = [ref.env.sim.get_state().flatten().copy()]
        write_json(run/'hold-start.json',dict(at=stamp(),pid=os.getpid(),steps_planned=20,source=str(source),code_commit=code_commit(),cpu_only=True))
        for step in range(20):
            guard.set_boundary(step,'precontact')
            with guard, GuardedIntegration(ref.env.sim,d,joint,lite_physics=ref.env.lite_physics,step=step,phase='precontact'):
                ref.env.step(action)
            states.append(ref.env.sim.get_state().flatten().copy())
        np.savez_compressed(run/'hold-states.npz',states=np.asarray(states),action=action)
        write_json(run/'hold-native-monitor.json',guard.save(run/'hold-native'))
        summary = dict(at=stamp(),steps=20,physical_substeps=500,task_outcomes=0,engineering_only=True,cpu_only=True,
            eef_drift_m=float(np.linalg.norm(d.site_xpos[site]-eef)),
            nonrobot_qpos_drift=float(np.max(abs(d.qpos[13:]-initial[13:]))),joint_monitor=joint.receipt(),
            opening=ref.trace()['target'],checker_success=bool(ref.env._check_success()))
        summary['passed'] = summary['eef_drift_m']<.01 and summary['nonrobot_qpos_drift']<1e-4 and not summary['checker_success']
        write_json(run/'stability-check.json',summary)
        print(summary,flush=True)
        assert summary['passed']
    finally:
        ref.env.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--batch',type=Path,required=True); parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args(); publish(args.batch,args.run)
