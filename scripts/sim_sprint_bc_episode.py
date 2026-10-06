"""Frozen BC network chunks through the native controller and feedback recorder."""
import argparse, copy, hashlib, json, os, pickle, random, shutil, subprocess, time, traceback
from collections import deque
from pathlib import Path
from datetime import datetime, timezone
import h5py, mujoco, numpy as np, torch
from robomimic.config import config_factory
import robomimic.utils.obs_utils as ObsUtils
from mobiwam.mobipi_checkpoint import load_policy_from_checkpoint
from mobiwam.mobipi_policy import sample_verified_future_chunk
from mobiwam.reference_formal_substep import FormalSubstepMonitor, FormalSafetyStop
from mobiwam.reference_prefix_safety import JointMarginMonitor, JointMarginStop, GuardedIntegration
from mobiwam.adapters.mobipi import _capture_controller_state, _restore_controller_state
from teleop_reference import Reference, write_json
from human_scene_pilot import restore_saved_integration
from mobiwam.task_video_identity import observe_native
from PIL import Image

def now():return datetime.now(timezone.utc).isoformat()
def load(p):return json.loads(Path(p).read_text())

def read_observation(ref,shape,step=0,action=None):
    """Checkpoint camera dimensions and the official robomimic RGB preprocessing.

    mujoco.Renderer returns top-first RGB, equivalent to official EnvRobosuite's
    bottom-first sim.render followed by [::-1]. No evidence camera is an input.
    """
    raw=dict(ref.env._get_observations(force_update=True));m,d=ref.model_data()
    raw['timesteps']=np.array([step]);raw['actions']=np.zeros(12) if action is None else np.asarray(action).copy()
    for camera in ref.policy_cameras:
        key=camera+'_image';_,height,width=shape['all_shapes'][key]
        renderer=mujoco.Renderer(m,height=height,width=width)
        try:
            renderer.update_scene(d,camera=camera,scene_option=ref.render_options)
            raw[key]=ObsUtils.process_obs(renderer.render().copy(),obs_key=key)
        finally:renderer.close()
    # RolloutPolicy's coordinate conversion also reads world EEF fields that
    # are absent from the network shape metadata. Preserve native wrapper keys.
    keys=set(shape['all_obs_keys'])|{k for k in raw if k.startswith('robot0_') and not k.endswith('proprio-state')}|{'actions','timesteps'}
    return {k:np.asarray(raw[k]).copy() for k in sorted(keys) if k!='lang_emb'}

def capture_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(x):
    random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch']);torch.cuda.set_rng_state_all(x['cuda'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--slot',type=int,required=True);ap.add_argument('--attempt',type=int,default=1)
    a=ap.parse_args();r=a.run;roster=load(r/'policy/ability-roster.json');slot=roster['slots'][a.slot-1]
    assert slot['split']=='train' and slot['group_id'] in {x['group_id'] for x in load(r/'data/reference-sources.json')}
    suffix='' if a.attempt==1 else f'-engineering-attempt-{a.attempt}'
    out=r/'policy'/f"ability-{a.slot:02d}-{slot['group_id']}{suffix}";out.mkdir(exist_ok=False)
    started=now();write_json(out/'process.json',dict(started_at=started,pid=os.getpid(),argv=__import__('sys').argv,python=__import__('sys').executable,
        code_commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip(),
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),MUJOCO_EGL_DEVICE_ID=os.environ.get('MUJOCO_EGL_DEVICE_ID')))
    src=Path(slot['source']);dest=out/src.name;dest.mkdir()
    for name in ('model.xml','integration.npy','ep_meta.json','rng.json','source.json','target-binding.json'):shutil.copy2(src/name,dest/name)
    shutil.copy2(src.parent/'env_config.json',out/'env_config.json')
    ref=None;attempt=None;queries=0;query_seconds=0.;status='engineering_failure';error=None;native=None;partial=False
    try:
        checkpoint=Path(roster['checkpoints'][slot['task']]);cfg_json=load(checkpoint.parents[1]/'config.json');cfg=config_factory(cfg_json['algo_name'])
        with cfg.values_unlocked():cfg.update(cfg_json)
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        import robomimic.utils.lang_utils as LangUtils
        LangUtils.PROJECT_HF_HOME=Path('/share/personal/chensiyu/haokaijiang/MobiWAM/cache/huggingface')
        model,policy,lang,env_meta,shape=load_policy_from_checkpoint(cfg,checkpoint)
        assert type(model).__name__=='BC_Transformer_GMM' and cfg.train.frame_stack==10
        write_json(out/'policy-binding.json',dict(task=slot['task'],checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            actual_policy_class=type(model).__name__,actual_policy_module=type(model).__module__,shape_metadata=shape,
            chunk_length=10,action_layout='right 0:6 gripper6 base7:10 torso10 mode11; official checkpoint normalization',
            policy_sampling_seed=20261006,observation_preprocessing='native top-first RGB, official ObsUtils.process_obs; checkpoint camera sizes',
            human_actions_used=False,reference_actions_used=False))
        ref=Reference(argparse.Namespace(output=str(out),task=slot['task'],layout=1,style=0,seed=slot['seed'],self_test=True,
            source=str(dest),replay_attempt=None,resume_attempt=None,width=1280,height=720))
        restore_saved_integration(ref);before=ref.integration().copy();m,d=ref.model_data()
        expected=load(dest/'target-binding.json')
        ref.identity_expected=dict(task=slot['task'],fixture_name=expected['fixture_name'],fixture_class=expected['fixture_class'],model_sha256=hashlib.sha256((dest/'model.xml').read_bytes()).hexdigest())
        ref.identity_context=dict(run_id=r.name,group_id=slot['group_id']);native=observe_native(ref,ref.identity_expected)
        assert not ref.env._check_success()
        random.seed(20261006);np.random.seed(20261006);torch.manual_seed(20261006);torch.cuda.manual_seed_all(20261006)
        policy.start_episode(lang=ref.env.get_ep_meta()['lang'])
        obs=read_observation(ref,shape);history={k:deque([v.copy() for _ in range(10)],maxlen=10) for k,v in obs.items()}
        assert np.max(abs(ref.integration()-before),initial=0)<=1e-6
        bundle=dict(integration=ref.integration().copy(),controller=_capture_controller_state(ref.env),history=copy.deepcopy(history),rng=capture_rng(),
            policy_init_version='policy-init-v1',initial_frame_padding=True,initial_frame_timestamps=[float(d.time)]*10,
            policy_buffers={k:copy.deepcopy(v) for k,v in vars(model).items() if 'buffer' in k or 'hidden' in k})
        with (out/'policy-init-v1.pkl').open('xb') as f:pickle.dump(bundle,f)
        # A real zero-action round trip of the new common boundary.
        mujoco.mj_setState(m,d,bundle['integration'],ref.kind);mujoco.mj_forward(m,d)
        scratch=mujoco.MjData(m);mujoco.mj_setState(m,scratch,bundle['integration'],ref.kind);d.qacc_warmstart[:]=scratch.qacc_warmstart
        _restore_controller_state(ref.env,bundle['controller']);history=copy.deepcopy(bundle['history']);restore_rng(bundle['rng'])
        after_obs=read_observation(ref,shape)
        diffs={k:float(np.max(abs(after_obs[k].astype(float)-obs[k].astype(float)),initial=0)) for k in obs}
        if max(diffs.values(),default=0)>1e-6:raise ValueError('policy-init restore input differs: '+str(diffs))
        write_json(out/'policy-init-restore.json',dict(created_at=now(),zero_actions=True,integration_max_abs_error=float(np.max(abs(ref.integration()-bundle['integration']),initial=0)),
            observation_field_diffs=diffs,initial_frame_padding=True,parent_source=str(src),native=native))
        # Reuse parent/native evidence camera; add an independent wider base view.
        parent_camera=load(dest/'source.json')['camera'];ref.evidence_camera=parent_camera
        ref.panoramic_camera=dict(parent_camera,distance=parent_camera['distance']*1.45)
        Image.fromarray(ref.frame(parent_camera)).save(out/'target-preview.jpg');Image.fromarray(ref.frame(ref.panoramic_camera)).save(out/'panoramic-preview.jpg')
        ref.route='E';ref.label='SIM-SPRINT-v1 autonomous frozen BC E';ref.begin(restore_source=False);attempt=ref.recording['path']
        guard=FormalSubstepMonitor(ref,native['fixture_name']);arm=ref.robot.part_controllers['right'];base=ref.robot.part_controllers['base']
        joints=[int(np.flatnonzero(m.jnt_qposadr==i)[0]) for i in arm.qpos_index]
        margin=JointMarginMonitor(arm.qpos_index,m.jnt_range[joints],[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) for j in joints])
        anchor=d.qpos[base.qpos_index].copy();chunk=None;offset=0;rows=[]
        with (out/'query-action-feedback.jsonl').open('x') as log:
            for step in range(2400):
                if offset==0:
                    stacked={k:np.stack(list(v)) for k,v in history.items()};t=time.monotonic()
                    with torch.no_grad():ev=sample_verified_future_chunk(policy,stacked)
                    query_seconds+=time.monotonic()-t;queries+=1;chunk=ev.chunk.copy()
                    if chunk.shape!=(10,12) or not np.isfinite(chunk).all():raise ValueError('invalid BC chunk')
                    np.savez_compressed(out/f'query-{queries:04d}.npz',chunk=chunk,official_first_action=ev.official_first_action)
                original=chunk[offset].copy();action=original.copy();action[7:10]=0.;action[11]=-1.
                guard.set_boundary(step,'manipulate');initial=ref.integration().copy()
                try:
                    with guard:
                        with GuardedIntegration(ref.env.sim,d,margin,lite_physics=ref.env.lite_physics,step=step,phase='manipulate'):
                            ref.step(action)
                except (FormalSafetyStop,JointMarginStop) as exc:
                    partial=True;np.savez_compressed(attempt/'partial-control-step.npz',initial_integration=initial,terminal_integration=ref.integration(),attempted_action=action)
                    write_json(attempt/'safety-stop.json',dict(at=now(),step=step,phase='manipulate',failure=exc.failure))
                    status='native_forbidden_contact_stop' if isinstance(exc,FormalSafetyStop) else 'joint_margin_stop'
                    if ref.recording:ref.finish(status)
                    break
                drift=float(np.max(abs(d.qpos[base.qpos_index]-anchor),initial=0))
                item=dict(step=step,query=queries,chunk_offset=offset,policy_raw=original.tolist(),mapped_action=action.tolist(),base_generalized_drift=drift,
                    sim_time=float(d.time),checker_success=bool(ref.env._check_success()))
                log.write(json.dumps(item)+'\n');log.flush();rows.append(item)
                if not ref.recording:status=load(attempt/'result.json')['reason'];break
                obs=read_observation(ref,shape,step+1,action)
                for k,v in obs.items():history[k].append(v.copy())
                offset=(offset+1)%10
                if step%50==0:print(json.dumps(dict(slot=a.slot,step=step,queries=queries,sim_time=float(d.time),progress=ref.trace()['target'])),flush=True)
            else:status='policy_budget_stop_120s';ref.finish(status)
        write_json(attempt/'formal-native-substeps-receipt.json',guard.save(attempt));write_json(attempt/'joint-margin-monitor.json',margin.receipt())
        result=load(attempt/'result.json')
        write_json(out/'completed.json',dict(started_at=started,ended_at=now(),slot=a.slot,group_id=slot['group_id'],task=slot['task'],route='E',
            status=status,native_success=result['checker_success'],steps=result['steps'],attempt=str(attempt),policy_queries=queries,
            query_seconds=query_seconds,partial_native_control=partial,clearance_qualification='pending_actual_swept_audit',
            human_intervention=False,reference_actions_used=False,state_injection_during_episode=False,formal_train_ready=False,
            controller='BC_Transformer_GMM official action chunk; base7:10=0 mode11=-1 E lock',independent_source_increment=0))
        print(json.dumps(load(out/'completed.json')),flush=True)
    except BaseException as exc:
        error=repr(exc);traceback.print_exc()
        write_json(out/'failure.json',dict(started_at=started,ended_at=now(),slot=a.slot,group_id=slot['group_id'],error=error,
            traceback=traceback.format_exc(),policy_queries=queries,attempt=str(attempt) if attempt else None,
            usable_outcome=bool(attempt and (attempt/'result.json').exists())))
        raise
    finally:
        if ref:
            if ref.recording:
                if ref.recording['n']:ref.finish('engineering_exception')
                else:
                    record=ref.recording
                    for key in ('h','trace','video','panoramic_video'):
                        if record.get(key) is not None:record[key].close()
                    write_json(record['path']/'result.json',dict(started_at=record['started'],ended_at=now(),steps=0,
                        checker_success=False,reason='engineering_exception_before_first_complete_action',usable_scientific_outcome=False))
                    ref.recording=None
            if ref.observation_renderer:ref.observation_renderer.close()
            if ref.renderer:ref.renderer.close()
            ref.env.close()

if __name__=='__main__':main()
