"""One selected feedback episode on an existing validation Source; never replay."""
import argparse,json,shutil,subprocess,os,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
import mujoco
import h5py
from PIL import Image
from teleop_reference import Reference,write_json
from reference_executor import run_route
from mobiwam.dr_v04_r4 import predict,SCOPE
from mobiwam.reference_plan_reuse import verify_plan_seal
from mobiwam.reference_dispatch import rejection_reason
from mobiwam.task_video_identity import observe_native
from mobiwam.dr_v04_r2 import prepare_camera
from mobiwam.reference_feature_interface import PROPRIO_KEYS

def load(p):return json.loads(Path(p).read_text())
def now():return datetime.now(timezone.utc).isoformat()

def main():
    a=argparse.ArgumentParser();a.add_argument("--run",type=Path,required=True);a.add_argument("--slot",type=int,required=True)
    a.add_argument("--attempt",type=int,default=1);args=a.parse_args()
    r=args.run;freeze=load(r/"frozen-six/roster.json");slot=freeze["slots"][args.slot-1];row=slot["scientific_row"]
    out=r/"episodes"/f'{args.slot:02d}-{slot["group_id"]}'/f'engineering-attempt-{args.attempt}'
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/"process.json",dict(started_at=now(),pid=os.getpid(),command=__import__("sys").argv,
      executable=__import__("sys").executable,code_commit=subprocess.check_output(["git","-C",str(Path(__file__).resolve().parents[1]),"rev-parse","HEAD"],text=True).strip(),
      CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),MUJOCO_EGL_DEVICE_ID=os.environ.get("MUJOCO_EGL_DEVICE_ID"),kind=SCOPE))
    prior=Path(row["plan_run"]);planning=load(row["candidate_features"]);source=Path(planning["source"])
    seal=verify_plan_seal(prior,source.name)
    assert row["split"]=="validation" and planning["source_integration_unchanged"]
    expected=np.load(prior/planning["restored_source_integration"],allow_pickle=False)
    dest=out/source.name;dest.mkdir()
    for name in ("model.xml","integration.npy","ep_meta.json","rng.json","source.json","target-binding.json"):
        shutil.copy2(source/name,dest/name)
    shutil.copy2(prior/"env_config.json",out/"env_config.json")
    cfg=load(out/"env_config.json");assert cfg["env_name"]==row["task"]
    ref=None
    try:
        ref=Reference(argparse.Namespace(output=str(out),task=cfg["env_name"],layout=1,style=0,seed=row["seed"],
                      self_test=True,source=str(dest),replay_attempt=None,resume_attempt=None,width=1920,height=1080))
        ref.restore();before=ref.integration().copy();error=float(np.max(abs(before-expected)))
        if error>1e-10:raise ValueError(f"Source integration restore differs {error}")
        if ref.env.rng.bit_generator.state!=load(source/"rng.json"):raise ValueError("Source RNG restore differs")
        ref.identity_expected=row;ref.identity_context=dict(run_id=r.name,group_id=row["group_id"])
        native=observe_native(ref,row)
        values=dict(ref.env._get_observations(force_update=True));diff={}
        with h5py.File(row["source_input_path"]) as f:
            obs=f["data/demo_0/obs"]
            for key in PROPRIO_KEYS:
                current=np.asarray(values[key]);old=np.asarray(obs[key][0])
                if current.shape!=old.shape or not np.isfinite(current).all():raise ValueError("cached proprio identity "+key)
                diff[key]=float(np.max(abs(current-old)))
        if max(diff.values())>1e-8:raise ValueError("cached Source proprio differs "+repr(diff))
        write_json(out/"cache-source-restore-receipt.json",dict(created_at=now(),source=str(source),copied_source=str(dest),
         source_restore_max_error=error,proprio_max_abs_diffs=diff,proprio_tolerance=1e-8,RNG_identical=True,
         native_identity=native,cached_source_input=row["source_input_path"],cached_source_input_sha256=row["source_input_sha256"],
         cached_candidate_features=row["candidate_features"],cached_context=row["context_path"],zero_task_actions=True,
         policy_camera_names=ref.policy_cameras,encoder_calls=0,semantic_scope="same native model/static policy cameras/physical initial state and proprio; cached first-frame embedding reused"))
        # Only independent free/evidence cameras change.
        camera=prepare_camera(ref,row,out/"camera-preview")
        ref.evidence_camera=camera["camera"];ref.apply_camera(ref.evidence_camera)
        eligible=[c for c in camera["candidates"] if c["minimum_visibility_ratio"]>=1.]
        panoramic=max(eligible,key=lambda c:c["pixels"]["base"])
        ref.panoramic_camera=panoramic["camera"]
        Image.fromarray(ref.frame(ref.panoramic_camera).copy()).save(out/"camera-preview/panoramic.jpg")
        ref.apply_camera(ref.evidence_camera)
        write_json(out/"camera-preview/panoramic-camera.json",dict(camera=ref.panoramic_camera,source_visible_pixels=panoramic["pixels"],
             zero_task_actions=True,independent_evidence_camera=True,policy_cameras_unchanged=ref.policy_cameras))
        if np.max(abs(ref.integration()-before))>1e-10:raise ValueError("camera preview changed Source")
        assert torch.cuda.is_available() and "A800" in torch.cuda.get_device_name(0)
        prediction,X,xs=predict(freeze,slot,torch.device("cuda:0"))
        write_json(out/"model-prediction.json",prediction);np.savez_compressed(out/"cached-inference-input.npz",X=X,scaled=xs)
        route=prediction["selection"]["route"]
        if route is None:raise ValueError("learned selector unsupported")
        if rejection_reason(planning,route,True) is not None:raise ValueError("frozen selected plan rejected")
        # Reuse immutable precomputed waypoints; no planner/old actions/LLM execution.
        points=load(prior/"waypoints.json")
        for p in points:
            for key in ("pos","rot","base","arm_qpos"):
                if key in p:p[key]=np.asarray(p[key],dtype=float)
        ref.dock_plan=load(prior/"dock-plan.json")
        assert ref.dock_plan["selected"]["id"]==planning["selected_dock_id"]
        ref.label="R4_existing_validation_Source_development_model_selected_feedback_execution"
        write_json(out/"dispatch.json",dict(created_at=now(),slot=args.slot,group_id=row["group_id"],route=route,
                   checkpoint_sha256=freeze["checkpoint_sha256"],input_source="identity-bound frozen Source cache",
                   execution_scope=SCOPE,Source_restore_error=error,formal_train_ready=False,horizon_steps=2400,
                   planning_code=seal["planning_code_commit"],old_actions_replayed=False,policy_generator="reference_executor.mapped_action with current native observations"))
        print(json.dumps(dict(dispatch="started",slot=args.slot,group=row["group_id"],route=route,pid=os.getpid())),flush=True)
        attempt=Path(run_route(ref,route,points,2400,execution_scope=SCOPE))
        result=load(attempt/"result.json");terminal=ref.trace()
        write_json(out/"terminal-live-state.json",dict(after=terminal,partial_control_step=(attempt/"partial-control-step.npz").is_file()))
        receipt=dict(ended_at=now(),slot=args.slot,group_id=row["group_id"],task=row["task"],route=route,attempt=str(attempt),
                     status="completed_development_outcome",result=result,primary_outcome=True,
                     selected_by_model=True,checkpoint_sha256=freeze["checkpoint_sha256"],model_prediction=str(out/"model-prediction.json"),
                     cache_restore=str(out/"cache-source-restore-receipt.json"),panoramic_video=str(attempt/"panoramic.mp4"),
                     original_video=str(attempt/"original.mp4"),native_task_manifest=str(attempt/"task-video-manifest.json"),
                     full_replay_performed=False,continuous_swept_audit_performed=False,formal_train_ready=False)
        write_json(out/"completed.json",receipt)
        print(json.dumps(dict(slot=args.slot,route=route,result=result["reason"],success=result["checker_success"],steps=result["steps"],video=receipt["original_video"])),flush=True)
    except BaseException as exc:
        write_json(out/"failure.json",dict(ended_at=now(),exception=repr(exc),traceback=traceback.format_exc(),slot=args.slot,
             outcome_artifacts=[str(p) for p in out.glob("source-*/*/attempt-*/result.json")],restore_or_recording_error=True))
        raise
    finally:
        if ref is not None:
            if ref.observation_renderer is not None:ref.observation_renderer.close()
            if ref.renderer is not None:ref.renderer.close()
            ref.env.close()

if __name__=="__main__":main()
