"""Zero-outcome new development preflight with the inherited frozen reference."""
import argparse,json,shutil,subprocess,time,traceback,os
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from teleop_reference import Reference,write_json
from human_scene_pilot import restore_saved_integration
from reference_geometry_v16 import plan_dock
from reference_planning import compile_candidates
from mobiwam.reference_transfer import compile_transferred_path

def load(p):return json.loads(Path(p).read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--group',required=True);a=ap.parse_args();r=a.run
    slots=load(r/'freeze.json')['main_slots'];slot=next(x for x in slots if x['group_id']==a.group)
    assert slot['split']=='development-validation'
    mirrors=load('/share/personal/chensiyu/haokaijiang/MobiWAM/artifacts/MMWAM-OBC-002-DR/DR-v0.4/20260930T115330Z-live-preflight/reference-mirror-receipt.json')['records']
    reference=Path(next(x for x in mirrors if x['task']==slot['task'])['mirror_attempt'])
    out=r/'reference-plans'/a.group;out.mkdir(parents=True,exist_ok=False);source=Path(slot['source']);dest=out/source.name;dest.mkdir()
    for name in ('model.xml','integration.npy','ep_meta.json','rng.json','source.json','target-binding.json'):shutil.copy2(source/name,dest/name)
    shutil.copy2(source.parent/'env_config.json',out/'env_config.json')
    ref=None;t0=time.monotonic()
    write_json(out/'process.json',dict(started_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),command=__import__('sys').argv,
        source=str(source),reference=str(reference),code_commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip(),
        input_rule='source and same inherited reference; no new outcome selection',route_proposal_caps=dict(E=1,D=5,A=5)))
    try:
        ref=Reference(argparse.Namespace(output=str(out),task=slot['task'],layout=1,style=0,seed=slot['environment_seed'],self_test=True,
            source=str(dest),replay_attempt=None,resume_attempt=None,width=1280,height=720))
        restore_saved_integration(ref);before=ref.integration().copy();ref.dock_proposal_cap=5
        points,transfer=compile_transferred_path(ref,reference);write_json(out/'transfer-receipt.json',transfer);write_json(out/'waypoints.json',points)
        ref.dock_plan=plan_dock(ref,points);assert len(ref.dock_plan['candidates'])<=5
        candidates=compile_candidates(ref,points,ref.dock_plan,out/'planning');write_json(out/'dock-plan.json',ref.dock_plan)
        error=float(np.max(abs(ref.integration()-before),initial=0));assert error<=1e-6
        write_json(out/'completed.json',dict(ended_at=datetime.now(timezone.utc).isoformat(),task=slot['task'],group_id=a.group,
            source=str(source),reference=str(reference),routes=[dict(route=x['route'],hard_valid=x['hard_valid']) for x in candidates['records']],
            restore_error=error,preflight_seconds=time.monotonic()-t0,actual_route_outcomes=0,formal_train_ready=False))
        print(json.dumps(load(out/'completed.json')),flush=True)
    except Exception as e:
        write_json(out/'failure.json',dict(ended_at=datetime.now(timezone.utc).isoformat(),exception=repr(e),traceback=traceback.format_exc(),
            preflight_seconds=time.monotonic()-t0,actual_route_outcomes=0,group_id=a.group));traceback.print_exc();raise
    finally:
        if ref:
            if ref.observation_renderer:ref.observation_renderer.close()
            if ref.renderer:ref.renderer.close()
            ref.env.close()

if __name__=='__main__':main()
