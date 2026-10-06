"""Zero-action native interface and one-demo target roundtrip diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from datetime import datetime, timezone
import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
from teleop_reference import Reference, write_json
from human_scene_pilot import restore_saved_integration
from mobiwam.task_video_identity import observe_native


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    candidates=json.loads((a.run/'data/policy-fit-candidates.json').read_text())
    x=candidates[0];source=Path(x['attempt']).parents[1]
    out=a.run/'preflight/native-interface';out.mkdir(exist_ok=False)
    ref=Reference(argparse.Namespace(output=str(out),task=x['task'],layout=1,style=0,seed=109,self_test=True,
        source=str(source),replay_attempt=None,resume_attempt=None,width=960,height=540))
    try:
        restore_saved_integration(ref);m,d=ref.model_data();before=ref.integration().copy()
        expected=json.loads((source/'target-binding.json').read_text())
        native=observe_native(ref,dict(task=x['task'],fixture_name=expected['fixture_name'],fixture_class=expected['fixture_class'],model_sha256=hashlib.sha256((source/'model.xml').read_bytes()).hexdigest()))
        arm=ref.robot.part_controllers['right'];arm.update(force=True)
        obs=ref.env._get_observations(force_update=True)
        details=dict(created_at=datetime.now(timezone.utc).isoformat(),zero_env_step=True,native=native,
            controller_attributes={k:np.asarray(getattr(arm,k)).tolist() for k in ('origin_pos','origin_ori','ref_pos','ref_ori_mat','input_max','input_min','output_max','output_min','qpos_index') if hasattr(arm,k)},
            input_ref_frame=arm.input_ref_frame,goal_update_mode=arm._goal_update_mode,
            obs={k:np.asarray(v).tolist() for k,v in obs.items() if k in ('robot0_base_pos','robot0_base_quat','robot0_base_to_eef_pos','robot0_base_to_eef_quat_site','robot0_eef_pos','robot0_eef_quat_site','robot0_gripper_qpos')})
        tests=[]
        with h5py.File(x['observations_path']) as f:
            g=f['data/demo_0'];actions=g['actions'][:]
            for i in np.linspace(0,len(actions)-1,min(30,len(actions)),dtype=int):
                ref.env.sim.set_state_from_flattened(g['states'][i]);ref.env.sim.forward()
                ref.robot.update_state();arm.update(force=True)
                original=actions[i,:6];arm.set_goal_update_mode('achieved');arm.set_goal(original)
                target_p=arm.goal_pos.copy();target_r=arm.goal_ori.copy()
                current_p=arm.world_to_origin_frame(arm.ref_pos)
                current_r=arm.goal_origin_to_eef_pose()[:3,:3]
                scaled=np.r_[target_p-current_p,Rotation.from_matrix(target_r@current_r.T).as_rotvec()]
                # scale_action is affine after clipping. Existing ranges are symmetric.
                inverse=scaled / ((np.asarray(arm.output_max)-np.asarray(arm.output_min))/2)
                arm.set_goal(inverse)
                error_p=float(np.max(abs(arm.goal_pos-target_p)))
                error_r=float(np.linalg.norm(Rotation.from_matrix(arm.goal_ori@target_r.T).as_rotvec()))
                tests.append(dict(timestep=int(i),original=original.tolist(),recovered=inverse.tolist(),pos_error_m=error_p,orientation_error_rad=error_r,
                    action_error=float(np.max(abs(original-inverse))),controller_origin_world_p=arm.origin_pos.tolist(),controller_origin_world_R=arm.origin_ori.tolist()))
        write_json(out/'nominal-target-roundtrip.json',dict(created_at=details['created_at'],original_demo=x['observations_path'],samples=tests,
            tolerance_m=1e-6,tolerance_rad=1e-6,tolerance_normalized_action=1e-5,
            passed=all(v['pos_error_m']<=1e-6 and v['orientation_error_rad']<=1e-6 and v['action_error']<=1e-5 for v in tests),
            env_steps=0,target='controller commanded goal, not achieved future displacement'))
        details['tests']=tests;write_json(out/'interface.json',details)
        restore_saved_integration(ref)
        camera=json.loads((source/'source.json').read_text())['camera']
        pano=dict(camera,distance=camera['distance']*1.45,azimuth=camera['azimuth']+25)
        Image.fromarray(ref.frame(camera).copy()).save(out/'original-main-preview.jpg')
        Image.fromarray(ref.frame(pano).copy()).save(out/'panorama-preview.jpg')
        write_json(out/'previews.json',dict(main=camera,panorama=pano,zero_env_step=True,same_frame=np.array_equal(np.asarray(Image.open(out/'original-main-preview.jpg')),np.asarray(Image.open(out/'panorama-preview.jpg')))))
        print(json.dumps(dict(native=expected['fixture_name'],roundtrip_passed=all(v['action_error']<=1e-5 for v in tests),controller_origin=details['controller_attributes'])),flush=True)
    finally:
        if ref.renderer:ref.renderer.close()
        ref.env.close()


if __name__=='__main__':main()
