"""Evaluate frozen v15 control on a separately specified development Source.

The reference remains immutable; only the initial state is supplied separately.
These small perturbations are excluded from all formal splits.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import traceback
from reference_executor_v15 import Reference, write_json, stamp, compile_path, plan_dock, run_route


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    p.add_argument('--source',required=True);p.add_argument('--routes',default='A');p.add_argument('--horizon',type=int,default=1400)
    args=p.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    attempt=Path(args.reference).resolve();original_source=attempt.parent.parent;source=Path(args.source).resolve()
    if (source/'model.xml').read_bytes()!=(original_source/'model.xml').read_bytes():
        raise ValueError('Start probe must preserve source model XML exactly')
    if json.loads((source.parent/'env_config.json').read_text())!=json.loads((original_source.parent/'env_config.json').read_text()):
        raise ValueError('Start probe environment config differs')
    # Isolated copy: recorder output must never be written inside human sources.
    dest=out/source.name;dest.mkdir()
    for name in ['model.xml','integration.npy','ep_meta.json','rng.json','source.json','target-binding.json']:
        shutil.copy2(source/name,dest/name)
    shutil.copy2(source.parent/'env_config.json',out/'env_config.json')
    write_json(out/'executor-spec.json',dict(created_at=stamp(),version='reference-feedback-v15',reference=str(attempt),
        source=str(source),routes=args.routes,horizon=args.horizon,source_sha256=hashlib.sha256((source/'integration.npy').read_bytes()).hexdigest(),
        scope='development_start_probe_only',independent_formal_source=False,policy_replacement=True,base_speed_caps=dict(CloseDrawer=.015,CloseSingleDoor=.09),translation_action_caps=dict(CloseDrawer=.10,CloseSingleDoor=.20),contact_model='slide-joint co-motion only under bilateral pad contact; other fixtures retain static guard',feedback='achieved OSC pose deltas; measured waypoint advancement; generalized-base position servo',
        limitations=['small perturbation of a human source, not held-out generalization','palm local linear guard, not full swept collision validator','no formal OBC feature export','D ranks 9 docks by sampled IK; navigation and between-sample collision validity not certified'],
        inputs='live simulator robot kinematics, current target joint and contact plus frozen demonstration geometry; no new-route future outcome input'))
    cfg=json.loads((out/'env_config.json').read_text())
    ref=Reference(argparse.Namespace(output=str(out),task=cfg['env_name'],layout=0,style=0,seed=7,self_test=True,
        source=str(dest),replay_attempt=None,resume_attempt=None,width=1920,height=1080))
    ref.label='autonomous_development_reference_feedback_v15'
    try:
        ref.restore();points=compile_path(ref,attempt);write_json(out/'waypoints.json',points)
        ref.dock_plan=plan_dock(ref,points);write_json(out/'dock-plan.json',ref.dock_plan)
        write_json(out/'recording-provenance.json',dict(data_kind='autonomous_development',version='reference-feedback-v15',formal_train_ready=False))
        results=[]
        for route in args.routes.split(','):
            if route not in ('A','E','D'):raise ValueError(route)
            if route=='D' and not ref.dock_plan['selected']['sampled_plan_valid']:
                write_json(out/'D-planning-rejected.json',dict(reason='no_sampled_ik_and_navigation_valid_dock',new_route_outcome=False));continue
            results.append(run_route(ref,route,points,args.horizon))
        write_json(out/'completed.json',dict(ended_at=stamp(),attempts=results))
    except BaseException:
        write_json(out/'error.json',dict(time=stamp(),traceback=traceback.format_exc()));raise
    finally:
        if ref.observation_renderer is not None:ref.observation_renderer.close()
        if ref.renderer is not None:ref.renderer.close()
        ref.env.close()

if __name__=='__main__':main()
