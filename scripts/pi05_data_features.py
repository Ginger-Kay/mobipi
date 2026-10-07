"""Bind fresh config RGB, frozen CLIP and geometry before primary outcomes."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from teleop_reference import Reference,write_json
from human_scene_pilot import restore_saved_integration
from mobiwam.extract_features import FrozenCLIPVisionEncoder,VISUAL_KEYS
from mobiwam.reference_feature_interface import PROPRIO_KEYS,source_context
from mobiwam.pi05_motion import docks,collaborative_paths
from mobiwam.scene004 import build_minimal_input,candidate_feature_vector,geometry_rule_select
from pi05_candidate_features import feature_record

def now():return datetime.now(timezone.utc).isoformat()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output-root',default='design/inputs');ap.add_argument('--freeze-output',default='design/features-freeze.json');a=ap.parse_args()
    model=Path('/share/personal/chensiyu/haokaijiang/MobiWAM/cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41')
    encoder=FrozenCLIPVisionEncoder(model,device='cuda:0',batch_size=3)
    design=json.loads((a.run/'design/start-design.json').read_text());completed=[]
    # Config order is static parent/task/split; no outcome is opened.
    for index,g in enumerate(design['selected']):
        out=a.run/a.output_root/g['config_id'];out.mkdir(parents=True,exist_ok=True)
        if (out/'features.json').exists():completed.append(json.loads((out/'features.json').read_text()));continue
        ref=Reference(argparse.Namespace(output=str(out/'native'),task=g['task'],layout=1,style=0,seed=g['environment_seed'],self_test=True,source=g['source'],replay_attempt=None,resume_attempt=None,width=640,height=360))
        try:
            restore_saved_integration(ref)
            def no_step(*args,**kwargs):raise AssertionError('features require zero env.step')
            ref.env.step=no_step;before=ref.integration().copy();m,d=ref.model_data()
            initial=json.loads((Path(g['source']).parent/'initial-state-controller.json').read_text())
            assert np.max(abs(d.qpos-np.asarray(initial['qpos'])))<=1e-6
            assert np.max(abs(d.qvel-np.asarray(initial['qvel'])))<=1e-6
            raw=ref.env._get_observations(force_update=True);sensors={key:np.asarray(raw[key])[None] for key in PROPRIO_KEYS}
            renderer=mujoco.Renderer(m,height=256,width=256)
            try:
                rgb={}
                for key in VISUAL_KEYS:
                    renderer.update_scene(d,camera=key[:-6],scene_option=ref.render_options)
                    image=renderer.render().copy();rgb[key]=image
                    sensors[key]=image.transpose(2,0,1)[None].astype(np.float32)/255.
                np.savez_compressed(out/'pre-outcome-RGB-sensors.npz',**rgb,**{key:np.asarray(raw[key]) for key in PROPRIO_KEYS})
                context=source_context(sensors,encoder);np.save(out/'context.npy',context)
            finally:renderer.close()
            assert np.max(abs(ref.integration()-before))<=1e-6
            plans={}
            for route,fn in [('D',docks),('A',collaborative_paths)]:
                try:plans[route]=fn(ref)
                except ValueError as exc:plans[route]=dict(primary=None,candidates=[],reason=str(exc),environment_step_calls=0)
                write_json(out/(route+'-plan.json'),plans[route])
            rows=[]
            for route in 'EDA':
                feature=feature_record(ref,route,plans.get(route),index)
                if feature is None:continue
                feature['derived']['planned_time_normalized']=feature['total_planned_time_s']/300.
                X=np.r_[build_minimal_input(context,candidate_feature_vector(feature['derived'])),[float(route==x) for x in 'EDA']].astype(np.float32)
                assert X.shape==(1048,) and np.isfinite(X).all();np.save(out/(route+'-X.npy'),X);rows.append(feature)
            receipt=dict(at=now(),parent_group=g['parent_group'],config_id=g['config_id'],family_id=g['family_id'],task=g['task'],role=g['role'],slot=g['slot'],tier=g['tier'],
                source=g['source'],routes=rows,geometry_selection=geometry_rule_select(rows),source_model_sha256=hashlib.sha256((Path(g['source'])/'model.xml').read_bytes()).hexdigest(),
                source_integration_sha256=hashlib.sha256((Path(g['source'])/'integration.npy').read_bytes()).hexdigest(),context_sha256=hashlib.sha256((out/'context.npy').read_bytes()).hexdigest(),
                RGB_source='this actual config restored state; no parent-anchor reuse',encoder=str(model),context_preprocessing='unchanged three CLIP L2-normalized vision tokens plus time-zero tanh proprio, mean4 into1024',
                route_candidates='existing planner primary geometric-cost candidate; no outcome selection',env_step_calls=0,policy_forward_calls=0,source_unchanged=True)
            write_json(out/'features.json',receipt);completed.append(receipt)
            print(json.dumps(dict(at=now(),config_id=g['config_id'],routes={x['route_family']:x['hard_valid'] for x in rows},configs=len(completed))),flush=True)
        finally:ref.env.close()
    write_json(a.run/a.freeze_output,dict(at=now(),configs=len(completed),records=completed,source_state_rgb_binding_verified=True,zero_policy_forward=True,zero_env_step=True,scaler_fit=False))

if __name__=='__main__':main()
