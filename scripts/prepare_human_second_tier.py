"""Outcome-blind second-tier scene drafts and zero-action readiness checks.

Never freezes a primary, runs a task action, or changes an existing Source.
"""
import argparse
import copy
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from prepare_human_scene_drafts import model, saved, obstacle_check, stamp, load, write
from mobiwam.reference_collision import SweptGeometry


def canonical_without_obstacle(xml):
    tree = ET.fromstring(xml)
    for parent in tree.iter():
        for child in list(parent):
            if child.get('name') == 'human_scene_obstacle_box':
                parent.remove(child)
    return ET.canonicalize(ET.tostring(tree, encoding='unicode'))


def replace_obstacle_position(xml, position):
    tree = ET.fromstring(xml)
    geoms = tree.findall(".//geom[@name='human_scene_obstacle_box']")
    assert len(geoms) == 1
    geoms[0].set('pos', ' '.join(map(str, position)))
    return ET.tostring(tree, encoding='unicode')


def second_tier(cfg):
    """One prespecified factor, no task-outcome search or random seed changes."""
    source = Path(cfg['source'])
    xml = (source/'model.xml').read_text()
    state = np.load(source/'integration.npy')
    m,d = model(xml,state)
    # Preserve the saved integration cache; mj_forward is not a physics step.
    mujoco.mj_setState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION)
    original = d.qpos.copy()
    binding = load(source/'target-binding.json')
    prefix,cat,_ = cfg['scene_id'].split('-')
    jid = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,binding['joints'][0]['name'])
    hid = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,binding['fixture_name']+('_door_handle' if prefix=='MW' else '_door_handle_handle'))
    bid = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'mobilebase0_base')
    assert min(jid,hid,bid)>=0
    inward=d.geom_xpos[hid]-d.xpos[bid];inward[2]=0;inward/=np.linalg.norm(inward)
    side=np.array([-inward[1],inward[0],0.])
    obstacle=copy.deepcopy(cfg['obstacle'])
    if cat=='O':
        joints=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name) for name in
                ['mobilebase0_joint_mobile_forward','mobilebase0_joint_mobile_side']]
        assert min(joints)>=0 and all(m.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE for j in joints)
        addresses=[int(m.jnt_qposadr[j]) for j in joints]
        axes=d.xaxis[joints,:2].T
        delta=np.linalg.solve(axes,.10*side[:2])
        d.qpos[addresses]+=delta
        factor=dict(name='base_lateral_offset_from_tier1',value1=0.,value2=.10,unit='m',world_delta=(.10*side).tolist(),allowed_qpos=addresses)
    elif cat in ('B','H'):
        if cat=='B': direction=side
        else:
            outward=-d.geom_xmat[hid].reshape(3,3)[:,1]
            direction=np.cross([0.,0.,1.],outward)
        old=float(obstacle['offset_m']);new=old-.05
        pos=np.array(obstacle['position'])+float(obstacle['sign'])*(new-old)*direction
        pos[2]=obstacle['position'][2]
        obstacle.update(position=pos.tolist(),offset_m=new)
        xml=replace_obstacle_position(xml,pos)
        factor=dict(name='base_side_offset' if cat=='B' else 'handle_side_offset',value1=old,value2=new,unit='m',allowed_qpos=[])
    else:
        assert cat=='M'
        address=int(m.jnt_qposadr[jid]);old=float(d.qpos[address]);d.qpos[address]=old*.75
        factor=dict(name='target_initial_joint_position',value1=old,value2=float(d.qpos[address]),unit='m' if m.jnt_type[jid]==mujoco.mjtJoint.mjJNT_SLIDE else 'rad',allowed_qpos=[address])
    newstate=saved(m,d)
    assert set(np.flatnonzero(d.qpos!=original)) <= set(factor['allowed_qpos'])
    assert canonical_without_obstacle(xml)==canonical_without_obstacle((source/'model.xml').read_text())
    nm,nd=model(xml,newstate)
    assert (m.nq,m.nv,m.nu)==(nm.nq,nm.nv,nm.nu)
    clearance=float(np.min(SweptGeometry(nm,target_prefix=binding['fixture_name']).distances(nd.qpos,'precontact')[1]))
    check=obstacle_check(nm,nd,binding['fixture_name'],jid) if obstacle else None
    if obstacle:obstacle.update(check)
    ready=clearance>=.0005 and (check is None or check['accepted'])
    return xml,newstate,obstacle,factor,dict(initial_static_clearance_m=clearance,obstacle=check,ready=ready)


def build(batch,run):
    rows=[]
    for category in ['O','B','H','M']:
        for prefix in ['MW','DR']:
            oldpath=batch/'scenes'/f'{prefix}-{category}-01'/'draft-v1/pilot.json'
            cfg=load(oldpath);scene=f'{prefix}-{category}-02'
            xml,state,obstacle,factor,check=second_tier(cfg)
            write(run/f'{scene}-design-check.json',check)
            if not check['ready']:
                rows.append(dict(scene_id=scene,status='static_rejected',check=check,factor=factor));continue
            root=batch/'scenes'/scene/'draft-v1';root.mkdir(parents=True,exist_ok=False)
            source=root/f'source-{run.name.split("-")[0]}-{scene}';source.mkdir()
            parent=Path(cfg['source'])
            (source/'model.xml').write_text(xml);np.save(source/'integration.npy',state)
            for name in ['rng.json','ep_meta.json']:shutil.copy2(parent/name,source/name)
            shutil.copy2(parent.parent/'env_config.json',root/'env_config.json')
            binding=load(parent/'target-binding.json');m,d=model(xml,state)
            j=binding['joints'][0];value=float(d.qpos[j['qpos_address']]);ratio=value/j['qpos'];j['qpos']=value
            binding['opening']['door']*=ratio;binding['verified_at']=stamp();write(source/'target-binding.json',binding)
            write(source/'source.json',dict(created_at=stamp(),grasp_command=False,camera=cfg['main_camera'],
                label='unfrozen_second_tier_human_scene',parent_source=str(parent),factor=factor,
                trace_status='Native task restore and binding check required before use'))
            new={**cfg,'scene_id':scene,'variant':2,'source':str(source),'created_at':stamp(),
                'target_binding':str(source/'target-binding.json'),'primary_enabled':False,'frozen_at':None,
                'frozen_source':None,'preview_visibility_pass':False,'restore_check':None,'restore_max_abs_error':None,
                'obstacle':obstacle,'main_factor':factor['name'],'factor':factor,
                'source_lineage':dict(parent_source=str(parent),family=cfg['scene_family_id'],same_environment=True,independent_source_increment=0),
                'initial_static_clearance_m':check['initial_static_clearance_m'],
                'intended_use':'Unfrozen human scene draft; no operator approval or automatic task labels',
                'code_commit':__import__('subprocess').check_output(['git','-C',str(Path(__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip()}
            # Preserve previously proposed balanced route order for this scene ID.
            import csv
            planned=next(r for r in csv.DictReader((batch/'scene-index.csv').open()) if r['scene_id']==scene)
            new['route_order']=list(planned['planned_order'])
            if obstacle:
                control=root/'no-added-obstacle-control';control.mkdir()
                (control/'model.xml').write_text(canonical_without_obstacle(xml));np.save(control/'integration.npy',state)
                for name in ['rng.json','ep_meta.json','source.json','target-binding.json']:shutil.copy2(source/name,control/name)
                new['control_snapshot']=str(control)
            else:new['control_snapshot']=str(source)
            write(root/'pilot.json',new)
            rows.append(dict(scene_id=scene,config=str(root/'pilot.json'),source=str(source),category=category,
                             status='draft_unfrozen',factor=factor,check=check))
            write(run/'second-tier.json',dict(at=stamp(),rows=rows,new_task_actions=0,independent_source_increment=0))
            print('BUILT',scene,check['initial_static_clearance_m'],flush=True)
    write(run/'second-tier.json',dict(at=stamp(),rows=rows,new_task_actions=0,independent_source_increment=0))


def restore_check(batch,run):
    from human_scene_pilot import PilotReference,make_args
    rows=[]
    for prefix in ['MW','DR']:
        configs=sorted((batch/'scenes').glob(f'{prefix}-[OBHM]-*/draft-v1/pilot.json'))
        first=load(configs[0]);ref=PilotReference(make_args(run/(prefix+'-environment'),first['task'],first['environment_seed'],Path(first['source'])),first)
        try:
            for path in configs:
                cfg=load(path);ref.pilot=cfg;ref.source=Path(cfg['source']);ref.restore();ref.bind()
                state=np.load(ref.source/'integration.npy')
                assert np.array_equal(state,ref.integration())
                assert not ref.env._check_success()
                actual_rng=ref.env.rng.bit_generator.state
                saved_rng=load(ref.source/'rng.json')
                # Reuse authoritative restore implementation and explicitly bind the original file.
                receipt=dict(scene_id=cfg['scene_id'],at=stamp(),restore_max_abs_error=ref.restore_receipt['max_abs_error'],
                    task=ref.args.task,fixture=ref.native['fixture_name'],opening=ref.native['opening'],
                    model_sha256=hashlib.sha256((ref.source/'model.xml').read_bytes()).hexdigest(),
                    restored_rng=actual_rng,saved_rng=saved_rng,zero_actions=True,checker_initial_success=False,
                    primary_enabled=False,operator_review='pending')
                rows.append(receipt);write(run/'restore-checks.json',dict(at=stamp(),rows=rows,new_physics_steps=0))
                print('RESTORED',cfg['scene_id'],receipt['restore_max_abs_error'],flush=True)
        finally:
            for name in ['renderer','observation_renderer']:
                obj=getattr(ref,name,None)
                if obj is not None:obj.close()
            ref.env.close()


def previews(batch,run):
    from PIL import Image
    from OpenGL import GL
    rows=[]
    for row in load(run/'second-tier.json')['rows']:
        if row['status']!='draft_unfrozen':continue
        cfg=load(Path(row['config']));source=Path(cfg['source'])
        for label,path in [('scene',source),('control',Path(cfg['control_snapshot']))]:
            if label=='control' and path==source:continue
            m=mujoco.MjModel.from_xml_path(str(path/'model.xml'));d=mujoco.MjData(m);state=np.load(path/'integration.npy')
            mujoco.mj_setState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION);warm=d.qacc_warmstart.copy();mujoco.mj_forward(m,d);d.qacc_warmstart[:]=warm
            assert np.array_equal(saved(m,d),state)
            out=source.parent/'preview' if label=='scene' else path/'preview';out.mkdir(exist_ok=False)
            with mujoco.Renderer(m,360,640) as renderer:
                option=mujoco.MjvOption();option.geomgroup[0]=0
                for name,key in [('main','main_camera'),('panoramic','panoramic_camera')]:
                    cam=mujoco.MjvCamera()
                    for k,v in cfg[key].items():
                        if k=='lookat':cam.lookat[:]=v
                        else:setattr(cam,k,v)
                    renderer.update_scene(d,camera=cam,scene_option=option);Image.fromarray(renderer.render()).save(out/(name+'.png'))
                device=GL.glGetString(GL.GL_RENDERER).decode();assert 'llvmpipe' in device.lower()
            rec=dict(scene_id=cfg['scene_id'],snapshot=label,source=str(path),preview=str(out),renderer=device,
                     physics_steps=0,restore_error=0,operator_visibility_approved=False)
            write(out/'receipt.json',rec);rows.append(rec);write(run/'preview-checks.json',rows)
            print('PREVIEW',cfg['scene_id'],label,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['build','restore','previews'])
    parser.add_argument('--batch',required=True,type=Path);parser.add_argument('--run',required=True,type=Path)
    args=parser.parse_args();{'build':build,'restore':restore_check,'previews':previews}[args.stage](args.batch,args.run)
