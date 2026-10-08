"""Frozen spatial starts and outcome-blind predictions; no fitting or dynamics."""
import argparse, hashlib, json, shutil, traceback
from pathlib import Path
from datetime import datetime, timezone
import mujoco
import numpy as np
from teleop_reference import Reference
from human_scene_pilot import restore_saved_integration
from mobiwam.adapters.mobipi import _capture_controller_state
from mobiwam.pi05_motion import arm_indices, docks, collaborative_paths
from mobiwam.reference_collision import SweptGeometry
from mobiwam.extract_features import FrozenCLIPVisionEncoder, VISUAL_KEYS
from mobiwam.reference_feature_interface import PROPRIO_KEYS, source_context
from mobiwam.scene004 import build_minimal_input, candidate_feature_vector, geometry_rule_select
from mobiwam.pi05_data_learning import Head, scale, decode, choose, SCALES
from pi05_candidate_features import feature_record
from pi05_drawer_pipeline import write as atomic_write, read, now, clean_json

def write(path,value):
    def arrays(x):
        if isinstance(x,np.ndarray):return arrays(x.tolist())
        if isinstance(x,dict):return {k:arrays(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [arrays(v) for v in x]
        return x
    atomic_write(path,arrays(value))

R=Path('/share/personal/chensiyu/haokaijiang/MobiWAM')
NAMES=('model.xml','integration.npy','ep_meta.json','rng.json','source.json','target-binding.json')
METHODS=['Fixed-E','OBC-MLP','Fixed-D','Geometry','Fixed-A','Train-best-fixed']
CORE=[(142,'far15'),(143,'far15'),(142,'left10'),(143,'right10'),(142,'yawL10'),(143,'yawR10')]
MAIN=[(142,'near05'),(143,'near05'),(142,'right10'),(143,'left10'),(142,'yawR10'),(143,'yawL10')]
EXTRA=['far05','far10','left05','left15','right05','right15']
PARAMS={'near05':('u',.05),'far15':('u',-.15),'left10':('l',.10),'right10':('l',-.10),
        'yawL10':('yaw',np.deg2rad(10)),'yawR10':('yaw',-np.deg2rad(10)),
        'far05':('u',-.05),'far10':('u',-.10),'left05':('l',.05),'left15':('l',.15),
        'right05':('l',-.05),'right15':('l',-.15)}

def definitions(r,v):
    p=r/'design/definitions-freeze.json'
    if p.exists():return read(p)
    previous=read(v/'design/start-design.json')['selected']
    parents={x['environment_seed']:x for x in previous if x['role']=='evaluation' and x.get('start_index')==0}
    assert set(parents)=={142,143}
    states=[]; jobs=[]
    order=CORE+MAIN+[(s,o) for o in EXTRA for s in (142,143)]
    for i,(s,o) in enumerate(order):
        old=parents[s];base={k:old[k] for k in ['parent_group','family_id','task','environment_seed']}
        base.update(config_id=old['config_id']+'-stress-'+o,parent_config_id=old['config_id'],original_source=old['source'],
                    offset_id=o,axis=PARAMS[o][0],magnitude=float(PARAMS[o][1]),role='evaluation',tier=1,slot=i+1,
                    state_index=i,block=i//2,panel='core' if i<6 else 'main_completion' if i<12 else 'extension',
                    known_development=True,blind_evaluation=False,template_status='seen-template')
        states.append(base)
        sequence=METHODS[i%6:]+METHODS[:i%6]
        for method in sequence:
            tag='stress-'+method
            path=r/'episodes'/tag/f"slot-{i+1:02d}-{base['config_id']}"/'engineering-attempt-0/completed.json'
            jobs.append(dict(base,key=f'{s}/{o}/{method}',method=method,tag=tag,split='evaluation',source='primary' if method.startswith('Fixed-') else 'fresh_online',
                             route=method[-1] if method.startswith('Fixed-') else None,hard_valid=None,status='not_prepared',
                             admitted=False,attempt=0,receipt=str(path),method_order=sequence.index(method)))
    freeze=dict(at=now(),status='frozen_before_any_new_rollout',states=states,methods=METHODS,jobs=jobs,
                original_baseline_only=True,offset_units='metres and radians',yaw_axis='world vertical about original native base root',
                policy_sampling_seed=20261008,train_best_fixed='E',geometry_rule='v1 unchanged',selector='v1 unchanged',
                model_receipts=str(v/'policy/final-evaluation-freeze.json'),recipe=str(v/'training/obc-recipe.json'),
                old_validation=str(v/'delivery/final-independent-verification.json'),sealed_tests_accessed=False,
                definitions_use_outcomes=False)
    assert len(states)==24 and len(jobs)==144 and len({j['key'] for j in jobs})==144
    assert [j['method'] for j in jobs[:6]]==METHODS
    write(p,freeze);write(r/'design/selected-plan.json',dict(at=now(),configurations=states,jobs=jobs))
    return freeze

def pose_transform(m,d,bid,bids,bdofs,delta,yaw):
    """Move only native base coordinates; compensate root pivot translation."""
    world=d.xpos[bid].copy();rotation=d.xmat[bid].reshape(3,3).copy()
    jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,bid)
    xy=jp[:2,bdofs[:2]].copy();yaw_response=float(jr[2,bdofs[2]])
    assert abs(np.linalg.det(xy))>1e-6 and abs(yaw_response)>1e-6
    d.qpos[bids[:2]]+=np.linalg.solve(xy,delta[:2]);d.qpos[bids[2]]+=yaw/yaw_response
    mujoco.mj_forward(m,d)
    residual=(world+delta-d.xpos[bid])[:2]
    mujoco.mj_jacBody(m,d,jp,jr,bid)
    d.qpos[bids[:2]]+=np.linalg.solve(jp[:2,bdofs[:2]],residual);mujoco.mj_forward(m,d)
    rz=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    assert np.max(abs(d.xpos[bid]-world-delta))<=1e-6
    assert np.max(abs(d.xmat[bid].reshape(3,3)-rz@rotation))<=1e-6
    return dict(base_body=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,bid),qpos_indices=bids.tolist(),qvel_indices=bdofs.tolist(),
                original_world_xyz=world.tolist(),original_rotation=rotation.tolist(),xy_jacobian=xy.tolist(),yaw_response=yaw_response,
                requested_world_delta=delta.tolist(),requested_world_yaw_rad=yaw,realized_world_xyz=d.xpos[bid].tolist(),
                realized_rotation=d.xmat[bid].reshape(3,3).tolist())

def contacts(m,d,fixture):
    result=[]
    for ct in d.contact[:d.ncon]:
        names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(x)) or '' for x in (ct.geom1,ct.geom2)]
        if any(fixture in n for n in names) and any('gripper' in n or 'finger' in n for n in names):result.append(sorted(names))
    return sorted(result)

def models(v):
    import torch
    available={};status={};prior=read(v/'policy/final-evaluation-freeze.json')
    for kind in ['MLP','Linear']:
        try:
            path=Path(prior['methods'][kind]['checkpoint']);state=torch.load(path,map_location='cpu',weights_only=False)
            head=Head(kind);head.load_state_dict(state['model']);head.eval();available[kind]=(head,state)
            status[kind]=dict(prior['methods'][kind],identity_receipt_reused=True,content_change_indication=False,
                              bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
        except Exception:status[kind]=dict(status='unavailable',reason=traceback.format_exc())
    try:available['ridge']=read(v/'training/tier-1/ridge/ridge.json');status['ridge']=dict(prior['methods']['ridge'],identity_receipt_reused=True)
    except Exception:status['ridge']=dict(status='unavailable',reason=traceback.format_exc())
    return available,status

def predict(available,xs,valid,geometry):
    import torch
    values={};choices={}
    for kind in ['MLP','Linear']:
        if kind not in available:choices[kind]='model_unavailable';continue
        model,state=available[kind];per={}
        for route,x in xs.items():
            with torch.inference_mode():raw=model(torch.from_numpy(scale(x[None],state['scaler']['mean'],state['scaler']['std']))).numpy()
            per[route]=decode(raw,state['heads'])[0].tolist()
        values[kind]=per;choices[kind]=choose(per,valid,state['heads'])
    if 'ridge' in available:
        ridge=available['ridge'];rc=ridge['recipe'];heads=rc['heads'];per={}
        for route,X in xs.items():
            x=scale(X[None],np.array(rc['mean']),np.array(rc['std']))[0];p=np.full(5,np.nan)
            for j,h in enumerate(heads):
                if h['status']=='constant':p[j]=h['constant']
                elif h['status']=='learnable':
                    fit=ridge['fits'][str(j)];value=float(x[1024:1045]@np.array(fit['coef'])+fit['intercept'])
                    p[j]=(np.clip(value,0,1) if j<3 else max(value,0))*SCALES[j]
            per[route]=p.tolist()
        values['ridge']=per;choices['ridge']=choose(per,valid,heads)
    else:choices['ridge']='model_unavailable'
    choices['Geometry']=geometry;choices['Train-best-fixed']='E' if valid.get('E') else 'X';choices['OBC-MLP']=choices['MLP']
    return dict(predictions=values,choices=choices)

def prepare(r,v):
    design=definitions(r,v);available,status=models(v);write(r/'policy/model-binding.json',dict(at=now(),models=status,fit_calls=0,scaler_refit=False))
    encoder=None
    try:
        encoder=FrozenCLIPVisionEncoder(R/'cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41',device='cuda:0',batch_size=3)
    except Exception:write(r/'preflight/encoder-unavailable.json',dict(at=now(),reason=traceback.format_exc()))
    rows=[];predictions=[]
    for g in design['states']:
        binding=r/'design/anchors'/g['config_id']/'binding.json';cache=r/'policy/predictions'/f"{g['config_id']}.json"
        if binding.exists() and cache.exists():rows.append(read(binding));predictions.append(read(cache));continue
        out=binding.parent;out.mkdir(parents=True,exist_ok=True);src=Path(g['original_source']);dest=out/src.name;dest.mkdir(exist_ok=True)
        for name in NAMES:shutil.copy2(src/name,dest/name)
        shutil.copy2(src.parent/'env_config.json',out/'env_config.json')
        ref=None;row=dict(g,source=str(dest),status='engineering_unavailable',hard_valid_routes={t:None for t in 'EDA'})
        prediction=dict(config_id=g['config_id'],status='engineering_unavailable',predictions={},choices={},valid_routes={t:None for t in 'EDA'})
        try:
            ref=Reference(argparse.Namespace(output=str(out/'native'),task='CloseDrawer',layout=1,style=0,seed=g['environment_seed'],self_test=True,source=str(dest),replay_attempt=None,resume_attempt=None,width=640,height=360))
            restore_saved_integration(ref);m,d=ref.model_data()
            ref.env.step=lambda *a,**k: (_ for _ in ()).throw(AssertionError('no dynamics in preparation'))
            expected=read(src.parent/'initial-state-controller.json')
            assert np.max(abs(d.qpos-np.asarray(expected['qpos'])))<=1e-6 and np.max(abs(d.qvel-np.asarray(expected['qvel'])))<=1e-6
            base=ref.robot.part_controllers['base'];bids=np.asarray(base.qpos_index);bdofs=np.asarray(base.qvel_index)
            bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,ref.base_body);fixture=ref.env.drawer
            target=read(dest/'target-binding.json');assert target['fixture_name']==fixture.name and target['fixture_class']=='Drawer'
            hid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,fixture.handle_name);assert hid>=0
            world=d.xpos[bid].copy();handle=d.geom_xpos[hid].copy();u=handle[:2]-world[:2];assert np.linalg.norm(u)>1e-6
            u/=np.linalg.norm(u);left=np.array([-u[1],u[0]]);q0=d.qpos.copy();vel=d.qvel.copy();before_contacts=contacts(m,d,fixture.name)
            delta=np.zeros(3);yaw=0.
            if g['axis']=='yaw':yaw=g['magnitude']
            else:delta[:2]=g['magnitude']*(u if g['axis']=='u' else left)
            mapping=pose_transform(m,d,bid,bids,bdofs,delta,yaw)
            other=np.setdiff1d(np.arange(m.nq),bids);assert np.max(abs(d.qpos[other]-q0[other]))<=1e-6 and np.max(abs(d.qvel-vel))<=1e-6
            mapping.update(handle_geom=fixture.handle_name,handle_center_world=handle.tolist(),u=u.tolist(),left=left.tolist(),
                           other_joint_max_diff=float(np.max(abs(d.qpos[other]-q0[other]))),velocity_max_diff=float(np.max(abs(d.qvel-vel))))
            np.save(dest/'integration.npy',ref.integration());info=read(dest/'source.json');info.update(created_at=now(),parent_source=str(src),config_id=g['config_id'],initializer='base-only frozen world radial/lateral/yaw transform; no IK compensation or dynamics',perturbation=mapping);write(dest/'source.json',info)
            qexpected=d.qpos.copy();restore_saved_integration(ref);m,d=ref.model_data();assert np.max(abs(d.qpos-qexpected))<=1e-6 and np.max(abs(d.qvel-vel))<=1e-6
            after_contacts=contacts(m,d,fixture.name);check=SweptGeometry(m,target_prefix=fixture.name,margin=.0005)
            _,distances=check.distances(d.qpos,'manipulate');clearance=float(np.min(distances,initial=.1));qids,_,limits=arm_indices(ref)
            margin=float(np.min(np.minimum(d.qpos[qids]-limits[:,0],limits[:,1]-d.qpos[qids])));reasons=[]
            if any(pair not in after_contacts for pair in before_contacts):reasons.append('original target contact broken by frozen transform')
            if margin<=.015:reasons.append('initial joint margin <=.015rad')
            if clearance<.0005:reasons.append('initial forbidden clearance <.0005m')
            if ref.env._check_success():reasons.append('initial native task already successful')
            write(out/'initial-state-controller.json',dict(at=now(),qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),controller=_capture_controller_state(ref.env),restore=ref.restore_receipt,env_step_calls=0,original_arm_gripper_fixture_preserved=True))
            row.update(mapping=mapping,joint_margin_rad=margin,initial_forbidden_clearance_m=clearance,original_target_contacts=before_contacts,new_target_contacts=after_contacts,rejection_reasons=reasons,
                       integration_sha256=hashlib.sha256((dest/'integration.npy').read_bytes()).hexdigest(),env_step_calls=0,policy_forward_calls=0)
            if reasons:
                row.update(status='X',hard_valid_routes={t:False for t in 'EDA'});prediction.update(status='X',choices={k:'X' for k in ['MLP','Linear','ridge','Geometry','Train-best-fixed','OBC-MLP']},valid_routes=row['hard_valid_routes'])
            else:
                row.update(status='bound');assert encoder is not None,'frozen encoder unavailable'
                inputs=r/'design/inputs'/g['config_id'];inputs.mkdir(parents=True,exist_ok=True);initial=ref.integration().copy()
                raw=ref.env._get_observations(force_update=True);sensors={key:np.asarray(raw[key])[None] for key in PROPRIO_KEYS};renderer=mujoco.Renderer(m,height=256,width=256)
                try:
                    rgb={}
                    for key in VISUAL_KEYS:
                        renderer.update_scene(d,camera=key[:-6],scene_option=ref.render_options);image=renderer.render().copy();rgb[key]=image;sensors[key]=image.transpose(2,0,1)[None].astype(np.float32)/255.
                    np.savez_compressed(inputs/'pre-outcome-RGB-sensors.npz',**rgb,**{k:np.asarray(raw[k]) for k in PROPRIO_KEYS});context=source_context(sensors,encoder);np.save(inputs/'context.npy',context)
                finally:renderer.close()
                plans={};features=[];xs={};valid={}
                for route,fn in [('D',docks),('A',collaborative_paths)]:
                    try:plans[route]=fn(ref)
                    except ValueError as exc:plans[route]=dict(primary=None,candidates=[],reason=str(exc))
                    write(inputs/(route+'-plan.json'),plans[route])
                for route in 'EDA':
                    f=feature_record(ref,route,plans.get(route),g['state_index']);valid[route]=bool(f and f['hard_valid'])
                    if f:
                        f['derived']['planned_time_normalized']=f['total_planned_time_s']/120.;x=np.r_[build_minimal_input(context,candidate_feature_vector(f['derived'])),[float(route==t) for t in 'EDA']].astype(np.float32)
                        assert x.shape==(1048,) and np.isfinite(x).all();np.save(inputs/(route+'-X.npy'),x);features.append(f)
                        if valid[route]:xs[route]=x
                assert np.max(abs(ref.integration()-initial))<=1e-6
                geometry=geometry_rule_select(features);geometry={x['candidate_id']:x['route_family'] for x in features}.get(geometry,'X')
                feature_receipt=dict(at=now(),config_id=g['config_id'],parent_group=g['parent_group'],source=str(dest),routes=features,geometry_selection=geometry,zero_env_step=True,zero_policy_forward=True,scaler_fit=False,RGB_source='this frozen configuration')
                write(inputs/'features.json',feature_receipt);row['hard_valid_routes']=valid;prediction.update(status='frozen',valid_routes=valid,**predict(available,xs,valid,geometry))
        except Exception:
            row.update(status='engineering_unavailable',reason=traceback.format_exc());prediction.update(status='engineering_unavailable',reason=row['reason'])
        finally:
            if ref:
                for name in ('renderer','pi05_renderer','observation_renderer'):
                    renderer=getattr(ref,name,None)
                    if renderer:renderer.close()
                ref.env.close()
        prediction.update(at=now(),state=row,models=status,outcomes_accessed=False);write(binding,row);write(cache,prediction);rows.append(row);predictions.append(prediction)
        plan=read(r/'design/selected-plan.json')
        for job in plan['jobs']:
            if job['config_id']!=g['config_id']:continue
            route=job['route'] if job['method'].startswith('Fixed-') else prediction['choices'].get(job['method'])
            job.update(route=route if route in ('E','D','A') else None,hard_valid=row['hard_valid_routes'].get(route) if route in ('E','D','A') else False if route=='X' else None,
                       status='engineering_unavailable' if row['status']=='engineering_unavailable' else 'unrun_model_unavailable' if route=='model_unavailable' else 'X_static_or_route_invalid' if route=='X' or row['hard_valid_routes'].get(route) is False else 'unrun',prediction_freeze=str(cache))
        write(r/'design/selected-plan.json',plan)
        write(r/'design/episode-roster.json',dict(at=now(),slots=rows+[dict(x,source=x['original_source']) for x in design['states'][len(rows):]]))
        print(json.dumps(dict(at=now(),config_id=g['config_id'],status=row['status'],valid=row['hard_valid_routes'],prepared=len(rows))),flush=True)
    write(r/'design/start-design.json',dict(at=now(),selected=rows,zero_env_step=True,zero_policy_forward=True))
    write(r/'policy/prediction-freeze.json',dict(at=now(),status='frozen_before_first_rollout',configurations=predictions,models=status,scaler_refit=False,outcomes_accessed=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--old-run',type=Path,required=True);ap.add_argument('--definitions-only',action='store_true');a=ap.parse_args()
    if a.definitions_only:definitions(a.run,a.old_run)
    else:prepare(a.run,a.old_run)
