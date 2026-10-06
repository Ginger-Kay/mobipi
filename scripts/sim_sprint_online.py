"""Live observation encoding, frozen selection and reference-only native feedback."""
import argparse,copy,hashlib,json,os,random,shutil,subprocess,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import h5py,mujoco,numpy as np,torch
from PIL import Image
from teleop_reference import Reference,write_json
from human_scene_pilot import restore_saved_integration
from reference_executor import run_route
from mobiwam.task_video_identity import observe_native
from mobiwam.extract_features import FrozenCLIPVisionEncoder,VISUAL_KEYS
from mobiwam.reference_feature_interface import PROPRIO_KEYS,source_context,assemble_input
from mobiwam.sim_sprint_learning import ActiveHead,scale,transform,select,ROUTES

ROOT=Path('/share/personal/chensiyu/haokaijiang/MobiWAM')
MODEL=ROOT/'cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41'
def load(p):return json.loads(Path(p).read_text())
def now():return datetime.now(timezone.utc).isoformat()

class SprintReference(Reference):
    def restore(self):return restore_saved_integration(self)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--group',required=True)
    ap.add_argument('--method',choices=['learned','geometry'],required=True);a=ap.parse_args();r=a.run
    assert load(r/'policy/readiness.json')['main_controller']=='reference_fallback'
    assert (r/'training/MLP/completed.json').exists() and (r/'comparison/final-prediction-receipt.json').exists()
    slot=next(x for x in load(r/'freeze.json')['main_slots'] if x['group_id']==a.group);assert slot['split']=='development-validation'
    bindings=load(r/'reference-plans/current-bindings.json') if (r/'reference-plans/current-bindings.json').exists() else {}
    planning_root=Path(bindings.get(a.group,str(r/'reference-plans'/a.group)));out=r/'episodes'/a.group/a.method;out.mkdir(parents=True,exist_ok=False)
    write_json(out/'process.json',dict(started_at=now(),pid=os.getpid(),argv=__import__('sys').argv,
        code_commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip(),
        controller='inherited reference feedback; frozen BC not ready; L1 only',method=a.method,group_id=a.group))
    if not (planning_root/'completed.json').exists():
        write_json(out/'not-executed.json',dict(at=now(),reason='reference_preflight_not_completed',source=str(slot['source']),
            scientific_episode_started=False,missing_slot_preserved=True));return
    plan=load(planning_root/'planning/candidate-features.json');eligible=[bool(x['hard_valid']) for x in plan['records']]
    if not any(eligible):
        write_json(out/'not-executed.json',dict(at=now(),reason='X_no_hard_valid_candidate',source=str(slot['source']),
            scientific_episode_started=False,missing_slot_preserved=True,planner_scope='simulator oracle geometry'));return
    source=Path(slot['source']);dest=out/source.name;dest.mkdir()
    for name in ('model.xml','integration.npy','ep_meta.json','rng.json','source.json','target-binding.json'):shutil.copy2(source/name,dest/name)
    shutil.copy2(source.parent/'env_config.json',out/'env_config.json');ref=None;episode_started=False
    try:
        ref=SprintReference(argparse.Namespace(output=str(out),task=slot['task'],layout=1,style=0,seed=slot['environment_seed'],self_test=True,
            source=str(dest),replay_attempt=None,resume_attempt=None,width=1280,height=720))
        restore_saved_integration(ref);before=ref.integration().copy();expected=np.load(source/'integration.npy');error=float(np.max(abs(before-expected),initial=0))
        assert error<=1e-6
        binding=load(source/'target-binding.json');ref.identity_expected=dict(task=slot['task'],fixture_name=binding['fixture_name'],
            fixture_class=binding['fixture_class'],model_sha256=hashlib.sha256((source/'model.xml').read_bytes()).hexdigest())
        ref.identity_context=dict(run_id=r.name,group_id=a.group,method=a.method);native=observe_native(ref,ref.identity_expected)
        random.seed(20261006);np.random.seed(20261006);torch.manual_seed(20261006);torch.cuda.manual_seed_all(20261006)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        raw=dict(ref.env._get_observations(force_update=True));m,d=ref.model_data();renderer=mujoco.Renderer(m,height=256,width=256)
        try:
            for camera in ref.policy_cameras:
                renderer.update_scene(d,camera=camera,scene_option=ref.render_options);raw[camera+'_image']=renderer.render().copy()
        finally:renderer.close()
        obs={k:raw[k].transpose(2,0,1)[None].astype(np.float32)/255. if k in VISUAL_KEYS else np.asarray(raw[k])[None].copy() for k in (*VISUAL_KEYS,*PROPRIO_KEYS)}
        torch.cuda.synchronize();t=time.monotonic();encoder=FrozenCLIPVisionEncoder(MODEL,device='cuda:0',batch_size=3)
        context=source_context(obs,encoder);torch.cuda.synchronize();encoder_seconds=time.monotonic()-t
        X=np.stack([np.r_[assemble_input(context,c['features']),[float(route==v) for v in ROUTES]] for route,c in zip(ROUTES,plan['records'])]).astype(np.float32)
        cp=torch.load(r/'training/MLP/step2000.pt',map_location='cpu');xs=scale(X,cp['scaler']['mean'],cp['scaler']['std'])
        torch.cuda.synchronize();t=time.monotonic()
        if a.method=='learned':
            model=ActiveHead('MLP',cp['active']).cuda().eval();model.load_state_dict(cp['model'])
            with torch.no_grad():raw_pred=model(torch.from_numpy(xs).cuda()).cpu().numpy()
            pred=transform(raw_pred,cp['head_support']);j=select(pred,eligible,cp['active'])
        else:
            def key(j):
                f=plan['records'][j]['features']
                return (-f['minimum_continuous_clearance_m'],-f['minimum_manipulability'],-f['minimum_joint_margin_rad'],
                    -f['minimum_policy_view_compatibility'],f['total_planned_base_path_m'],f['planned_time_normalized'],j)
            j=min([j for j in range(3) if eligible[j]],key=key);raw_pred=None;pred=None
        torch.cuda.synchronize();selection_seconds=time.monotonic()-t
        if j is None:raise ValueError('selector unsupported')
        route=ROUTES[j];np.savez_compressed(out/'live-inference-input.npz',X=X,scaled=xs,context=context,
            **({'raw_prediction':raw_pred,'prediction':pred} if pred is not None else {}))
        assert np.max(abs(ref.integration()-before),initial=0)<=1e-6
        write_json(out/'selection.json',dict(at=now(),method=a.method,selected_route=route,source_restore_error=error,native_identity=native,
            encoder_seconds=encoder_seconds,selector_seconds=selection_seconds,precomputed_planner_seconds=load(planning_root/'completed.json')['preflight_seconds'],
            live_scene_inputs=True,network_forwards=1 if a.method=='learned' else 0,old_actions_replayed=False,
            controller='reference_executor.mapped_action native feedback; no frozen BC execution',snapshot_init='current pose controller goals, unchanged physical Source',
            hard_valid_candidates=eligible,model_checkpoint=str(r/'training/MLP/step2000.pt'),formal_train_ready=False))
        # Existing accepted human evidence camera is independent from all policy inputs.
        human_cfg=load(slot['config']);ref.evidence_camera=human_cfg['main_camera'];ref.panoramic_camera=human_cfg['panoramic_camera']
        Image.fromarray(ref.frame(ref.evidence_camera)).save(out/'target-preview.jpg');Image.fromarray(ref.frame(ref.panoramic_camera)).save(out/'panoramic-preview.jpg')
        points=load(planning_root/'waypoints.json')
        for p in points:
            for key in ('pos','rot','base','arm_qpos'):
                if key in p:p[key]=np.asarray(p[key],dtype=float)
        ref.dock_plan=load(planning_root/'dock-plan.json');ref.label=f'SIM-SPRINT-v1 reference fallback {a.method} development'
        # Selector construction consumes Torch RNG; execution starts from one
        # declared common RNG boundary for both methods, plus Source env RNG.
        random.seed(20261006);np.random.seed(20261006);torch.manual_seed(20261006);torch.cuda.manual_seed_all(20261006)
        episode_started=True;t=time.monotonic()
        attempt=Path(run_route(ref,route,points,2400,execution_scope='SIM-SPRINT-v1_reference_fallback'))
        result=load(attempt/'result.json');terminal=ref.trace();write_json(out/'terminal-live-state.json',dict(after=terminal))
        write_json(out/'completed.json',dict(ended_at=now(),group_id=a.group,task=slot['task'],method=a.method,route=route,attempt=str(attempt),
            result=result,wall_execution_seconds=time.monotonic()-t,terminal_duration_s=float(terminal['sim_time']-before[0]),
            native_success=result['checker_success'],safety_qualification='pending_native_swept_audit',human_intervention=False,
            controller='reference feedback',frozen_BC_used=False,state_injection_during_episode=False,formal_train_ready=False))
        print(json.dumps(load(out/'completed.json')),flush=True)
    except Exception as e:
        write_json(out/'failure.json',dict(ended_at=now(),exception=repr(e),traceback=traceback.format_exc(),scientific_episode_started=episode_started,
            valid_outcome_files=[str(p) for p in out.glob('source-*/*/attempt-*/result.json')],group_id=a.group,method=a.method));traceback.print_exc();raise
    finally:
        if ref:
            if ref.recording:ref.finish('engineering_exception')
            if ref.observation_renderer:ref.observation_renderer.close()
            if ref.renderer:ref.renderer.close()
            ref.env.close()

if __name__=='__main__':main()
