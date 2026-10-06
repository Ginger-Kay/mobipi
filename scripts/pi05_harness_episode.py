"""Real query/action/env feedback E development episode, no teacher actions."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import os
import shutil
import subprocess
import time
import traceback
import http.client
import mujoco
import numpy as np
from PIL import Image
from teleop_reference import Reference,write_json
from human_scene_pilot import restore_saved_integration
from mobiwam.task_video_identity import observe_native
from mobiwam.reference_formal_substep import FormalSubstepMonitor,FormalSafetyStop
from mobiwam.reference_prefix_safety import JointMarginMonitor,JointMarginStop,GuardedIntegration
from mobiwam.adapters.mobipi import _capture_controller_state
from mobiwam.pi05_adapter import observation,query,reset,execute_static,CAMERAS


def now():return datetime.now(timezone.utc).isoformat()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--slot',type=int,required=True)
    p.add_argument('--checkpoint-step',type=int,required=True);p.add_argument('--port',type=int,default=8865);p.add_argument('--attempt',type=int,default=0);a=p.parse_args()
    roster=json.loads((a.run/'policy/policy-dev-roster.json').read_text());slot=roster['slots'][a.slot-1]
    out=a.run/'episodes'/f'policy-dev-step-{a.checkpoint_step}'/f"slot-{a.slot:02d}-{slot['config_id']}"/f'engineering-attempt-{a.attempt}'
    if a.attempt:
        for old in out.parent.glob('engineering-attempt-*/completed.json'):
            if json.loads(old.read_text()).get('usable_scientific_outcome'):raise ValueError('slot already has a usable outcome; retry prohibited')
        for old in out.parent.glob('engineering-attempt-*/failure.json'):
            if json.loads(old.read_text()).get('usable_outcome'):raise ValueError('slot has an executed prefix; retain unknown, no blind same-version repeat')
    out.mkdir(parents=True,exist_ok=False);started=now();wall_start=time.monotonic();src=Path(slot['source']);source=out/src.name;source.mkdir()
    for name in ('model.xml','integration.npy','ep_meta.json','rng.json','source.json','target-binding.json'):shutil.copy2(src/name,source/name)
    shutil.copy2(src.parent/'env_config.json',out/'env_config.json')
    write_json(out/'process.json',dict(started_at=started,pid=os.getpid(),argv=__import__('sys').argv,python=__import__('sys').executable,code_commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip(),
        task=slot['task'],route='E',parent_group=slot['parent_group'],config_id=slot['config_id'],family_id=slot['family_id'],policy_sampling_seed=20261006,evaluation_seed=20261006,environment_seed=slot['environment_seed']))
    ref=None;attempt=None;guard=None;margin=None;queries=0;steps=0;query_seconds=0.;status='engineering_unknown'
    try:
        connection=http.client.HTTPConnection('127.0.0.1',a.port,timeout=20);connection.request('GET','/status');response=connection.getresponse();binding=json.loads(response.read());connection.close()
        assert response.status==200 and Path(binding['checkpoint']).name==str(a.checkpoint_step)
        write_json(out/'policy-binding.json',dict(binding,camera_slot_mapping=CAMERAS,reference_actions_used=False))
        ref=Reference(argparse.Namespace(output=str(out),task=slot['task'],layout=1,style=0,seed=slot['environment_seed'],self_test=True,source=str(source),replay_attempt=None,resume_attempt=None,width=960,height=540))
        restore_saved_integration(ref);m,d=ref.model_data();expected=json.loads((source/'target-binding.json').read_text())
        ref.identity_expected=dict(task=slot['task'],fixture_name=expected['fixture_name'],fixture_class=expected['fixture_class'],model_sha256=hashlib.sha256((source/'model.xml').read_bytes()).hexdigest())
        ref.identity_context=dict(run_id=a.run.name,group_id=slot['config_id']);native=observe_native(ref,ref.identity_expected)
        assert not ref.env._check_success();before=ref.integration().copy();inputs,anchor=observation(ref)
        assert np.max(abs(ref.integration()-before))<=1e-6
        write_json(out/'initialization.json',dict(native=native,restore=ref.restore_receipt,zero_env_step=True,source_parent=str(src),initial_time=float(d.time),controller=_capture_controller_state(ref.env),policy_rng='fresh20261006; no artificial history'))
        primary=dict(lookat=d.site_xpos[ref.robot.eef_site_id['right']].tolist(),distance=1.55,azimuth=180 if slot['task']=='CloseDrawer' else 90,elevation=-12)
        camera=json.loads((source/'source.json').read_text())['camera'];panorama=dict(camera,distance=camera['distance']*1.3,azimuth=camera['azimuth']+25)
        first=ref.frame(primary).copy();second=ref.frame(panorama).copy();assert not np.array_equal(first,second)
        Image.fromarray(first).save(out/'target-preview.jpg');Image.fromarray(second).save(out/'panorama-preview.jpg')
        write_json(out/'camera-plan.json',dict(primary=primary,panorama=panorama,distinct_zero_action_rgb=True,main_readability='predeclared task-specific side/front plan; exact per Source preview kept; human review pending'))
        ref.apply_camera(primary);ref.panoramic_camera=panorama;ref.route='E';ref.label='PI05-HARNESS-v1 autonomous frozen pi05 E';reset(a.port)
        base=ref.robot.part_controllers['base'];base_target=d.qpos[base.qpos_index].copy();arm=ref.robot.part_controllers['right']
        joints=[int(np.flatnonzero(m.jnt_qposadr==i)[0]) for i in arm.qpos_index]
        margin=JointMarginMonitor(arm.qpos_index,m.jnt_range[joints],[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) for j in joints])
        ref.begin(restore_source=False);attempt=ref.recording['path'];guard=FormalSubstepMonitor(ref,native['fixture_name'])
        deadline=wall_start+1200;start_time=float(d.time);max_drift=0.;raw=None
        with (out/'query-action-feedback.jsonl').open('x') as log:
            for step in range(2400):
                if time.monotonic()>deadline:status='compute-timeout';ref.finish(status);break
                if step%5==0:
                    inputs,anchor=observation(ref);answer=query(a.port,inputs);raw=answer['actions'];queries+=1;query_seconds+=float(answer['query_seconds'])
                    np.savez_compressed(out/f'query-{queries:04d}.npz',**inputs,**answer,base_world_p=anchor['base_world_p'],base_world_R=anchor['base_world_R'],query_sim_time=anchor['sim_time'])
                    assert raw.shape==(10,32) and np.isfinite(raw).all()
                actual,point=execute_static(ref,raw[step%5],anchor,base_target);guard.set_boundary(step,'manipulate');initial=ref.integration().copy()
                try:
                    with guard:
                        with GuardedIntegration(ref.env.sim,d,margin,lite_physics=ref.env.lite_physics,step=step,phase='manipulate'):ref.step(actual)
                except (FormalSafetyStop,JointMarginStop) as exc:
                    np.savez_compressed(attempt/'partial-control-step.npz',initial_integration=initial,terminal_integration=ref.integration(),attempted_action=actual)
                    write_json(attempt/'safety-stop.json',dict(at=now(),step=step,failure=exc.failure));status='native_forbidden_contact_stop' if isinstance(exc,FormalSafetyStop) else 'joint_margin_stop';ref.finish(status);break
                steps+=1;drift=float(np.max(abs(d.qpos[base.qpos_index]-base_target)));max_drift=max(max_drift,drift)
                row=dict(step=step,query=queries,chunk_offset=step%5,policy_raw_normalized=answer['normalized'][step%5].tolist(),policy_nominal=raw[step%5].tolist(),world_target=dict(pos=point['pos'].tolist(),rot=point['rot'].tolist(),grasp=point['grasp']),actual_action=actual.tolist(),base_locked_target=base_target.tolist(),base_generalized_drift=drift,sim_time=float(d.time),checker_success=bool(ref.env._check_success()))
                log.write(json.dumps(row)+'\n');log.flush()
                if step%20==0:print(json.dumps(dict(slot=a.slot,step=step,queries=queries,checker_success=row['checker_success'],base_drift=drift)),flush=True)
                if not ref.recording:status=json.loads((attempt/'result.json').read_text())['reason'];break
            else:status='policy_budget_stop_120s';ref.finish(status)
        write_json(attempt/'formal-native-substeps-receipt.json',guard.save(attempt));write_json(attempt/'joint-margin-monitor.json',margin.receipt())
        result=json.loads((attempt/'result.json').read_text());write_json(out/'completed.json',dict(started_at=started,ended_at=now(),slot=a.slot,checkpoint_step=a.checkpoint_step,task=slot['task'],route='E',parent_group=slot['parent_group'],config_id=slot['config_id'],family_id=slot['family_id'],status=status,attempt=str(attempt),native_success=result['checker_success'],steps=result['steps'],policy_queries=queries,query_seconds=query_seconds,base_drift_max_generalized=max_drift,usable_scientific_outcome=bool(steps or (attempt/'partial-control-step.npz').exists()),safety_status='pending_actual_sweep',reference_actions_used=False,human_intervention=False,world_target_source='pi05 only',state_injection_during_episode=False,formal_train_ready=False))
        print((out/'completed.json').read_text(),flush=True)
    except BaseException:
        traceback.print_exc();write_json(out/'failure.json',dict(at=now(),traceback=traceback.format_exc(),attempt=str(attempt) if attempt else None,completed_control_steps=steps,usable_outcome=bool(steps)))
        if ref and ref.recording:ref.finish('engineering_exception')
        if guard and attempt:write_json(attempt/'formal-native-substeps-receipt.json',guard.save(attempt))
        if margin and attempt:write_json(attempt/'joint-margin-monitor.json',margin.receipt())
        raise
    finally:
        if ref:
            for name in ('renderer','observation_renderer','pi05_renderer'):
                renderer=getattr(ref,name,None)
                if renderer:renderer.close()
            ref.env.close()


if __name__=='__main__':main()
