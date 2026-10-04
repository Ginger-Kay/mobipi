"""Eight unfrozen human scene drafts. Static design only; no task outcomes."""
import argparse, json, shutil, itertools
from pathlib import Path
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from mobiwam.reference_collision import SweptGeometry

def stamp(): return datetime.now(timezone.utc).isoformat()
def write(path, value): path.write_text(json.dumps(value,indent=2,default=lambda x:x.tolist())+'\n')
def load(path): return json.loads(path.read_text())
def model(xml,state):
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
    mujoco.mj_setState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
    return m,d
def saved(m,d):
    x=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(m,d,x,mujoco.mjtState.mjSTATE_INTEGRATION);return x
def box_xml(xml,position,size):
    tree=ET.fromstring(xml);world=tree.find('worldbody')
    body=ET.SubElement(world,'body',name='human_scene_obstacle',pos=' '.join(map(str,position)))
    # One geom serves visual and collision representations; no hidden collision edits.
    ET.SubElement(body,'geom',name='human_scene_obstacle_box',type='box',size=' '.join(map(str,size)),
                  contype='1',conaffinity='1',group='1',rgba='.85 .25 .12 1',friction='1 .005 .0001')
    return ET.tostring(tree,encoding='unicode')
def obstacle_check(m,d,target,joint):
    g=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'human_scene_obstacle_box')
    # Native GJK/EPA only on an independent geometric checker, never physics model.
    import copy
    gm=copy.copy(m);gm.opt.enableflags|=int(mujoco.mjtEnableBit.mjENBL_NATIVECCD)
    gd=mujoco.MjData(gm);gd.qpos[:]=d.qpos;mujoco.mj_forward(gm,gd)
    ids=[i for i in range(m.ngeom) if i!=g and (m.geom_contype[i] or m.geom_conaffinity[i])]
    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(m.ngeom)]
    # Floor support is expected for the static column. Every other existing geom is checked.
    ids=[i for i in ids if 'floor' not in names[i]]
    vals=[mujoco.mj_geomDistance(gm,gd,g,i,.2,None) for i in ids]
    closest=int(np.argmin(vals));minimum=float(vals[closest])
    targets=[i for i in ids if names[i].startswith(target)]
    qid=m.jnt_qposadr[joint];q0=float(d.qpos[qid]);limit=0.
    # 101 fixed configurations plus a conservative intervening displacement bound.
    # Drawer translates; microwave target geometry radius <= 2m from hinge.
    bound=abs(q0-limit)/100*(1 if m.jnt_type[joint]==mujoco.mjtJoint.mjJNT_SLIDE else 2.)/2
    swept=.2
    for q in np.linspace(q0,limit,101):
        gd.qpos[qid]=q;mujoco.mj_forward(gm,gd)
        swept=min(swept,min(mujoco.mj_geomDistance(gm,gd,g,i,.2,None) for i in targets))
    return dict(initial_obstacle_clearance_m=minimum,closest_geom=names[ids[closest]],
                target_sweep_sample_min_m=float(swept),target_sweep_lower_bound_m=float(swept-bound),
                target_sweep_samples=101,accepted=minimum>=.005 and swept-bound>=.0005)
def publish(batch,run,cfg,base_xml,state,category,obstacle,index):
    scene=cfg['scene_id'].split('-')[0]+'-'+category+'-01'
    root=batch/'scenes'/scene/'draft-v1';root.mkdir(parents=True,exist_ok=False)
    source=root/('source-'+run.name.split('-')[0]+'-'+scene);source.mkdir()
    xml=box_xml(base_xml,obstacle['position'],obstacle['half_size']) if obstacle else base_xml
    m,d=model(xml,state)
    (source/'model.xml').write_text(xml);np.save(source/'integration.npy',state)
    parent=Path(cfg['source'])
    for name in ['ep_meta.json','rng.json','target-binding.json']:
        shutil.copy2(parent/name,source/name)
    shutil.copy2(parent.parent/'env_config.json',root/'env_config.json')
    binding=load(source/'target-binding.json');j=binding['joints'][0];q=float(d.qpos[j['qpos_address']])
    ratio=q/j['qpos'];j['qpos']=q;binding['opening']['door']*=ratio;binding['verified_at']=stamp()
    write(source/'target-binding.json',binding)
    write(source/'source.json',dict(created_at=stamp(),grasp_command=False,camera=cfg['main_camera'],
          label='unfrozen_common_stowed_start_scene_draft',parent_source=str(parent),trace_status='Recomputed by native environment restore verification before use'))
    distances=SweptGeometry(m,target_prefix=binding['fixture_name']).distances(d.qpos,'precontact')[1]
    clearance=float(np.min(distances));assert clearance>=.0005,(scene,clearance)
    order=['EDA','DAE','AED','EAD','DEA','ADE'][index%6]
    new={**cfg,'scene_id':scene,'config_version':'draft-v1','source':str(source),'created_at':stamp(),
         'primary_enabled':False,'frozen_at':None,'frozen_source':None,'route_order':list(order),
         'category':category,'variant':1,'target_binding':str(source/'target-binding.json'),
         'intended_use':'unfrozen human development scene; operator preview/practice and E/D/A checks required before primary',
         'preview_visibility_pass':False,'restore_max_abs_error':None,'restore_check':None,
         'paired_protocol_version':'human-eda-v2-stowed','common_stow_qpos':d.qpos[4:11].tolist(),
         'obstacle':obstacle,'main_factor':{'O':'no added obstacle','B':'base-side obstacle offset','H':'handle-side column offset','M':'initial target opening with fixed handle-side column'}[category],
         'source_lineage':dict(parent_source=str(parent),family=cfg['scene_family_id'],same_environment=True,
                              independent_source_increment=0,reason='Common stowed start and versioned obstacle/opening draft'),
         'initial_static_clearance_m':clearance,'new_task_actions':0,'formal_train_ready':False}
    if obstacle:
        control=root/'no-added-obstacle-control';control.mkdir();(control/'model.xml').write_text(base_xml);np.save(control/'integration.npy',state)
        for name in ['ep_meta.json','rng.json','source.json','target-binding.json']:shutil.copy2(source/name,control/name)
        new['control_snapshot']=str(control)
    else:new['control_snapshot']=str(source)
    write(root/'pilot.json',new)
    return dict(scene_id=scene,config=str(root/'pilot.json'),source=str(source),category=category,
                route_order=order,initial_static_clearance_m=clearance,obstacle=obstacle,status='draft_unfrozen')
def prepare(batch,run):
    rows=[]
    for task_index,prefix in enumerate(['MW','DR']):
        cfg=load(batch/'scenes'/(prefix+'-PILOT')/'pilot-v2-precontact/pilot.json')
        parent=Path(cfg['source']);xml=(parent/'model.xml').read_text();state=np.load(parent/'integration.npy')
        m,d=model(xml,state)
        old=load(batch/'scenes'/(prefix+'-PILOT')/'pilot-v1/pilot.json');om,od=model((Path(old['source'])/'model.xml').read_text(),np.load(Path(old['source'])/'integration.npy'))
        d.qpos[4:11]=od.qpos[4:11];d.qvel[:13]=0;d.qacc_warmstart[:13]=0;mujoco.mj_forward(m,d);d.qacc_warmstart[:13]=0
        state=saved(m,d)
        binding=load(parent/'target-binding.json');target=binding['fixture_name'];joint=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,binding['joints'][0]['name'])
        hname=target+('_door_handle' if prefix=='MW' else '_door_handle_handle')
        hid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,hname);h=d.geom_xpos[hid].copy()
        base=d.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'mobilebase0_base')].copy()
        inward=h-base;inward[2]=0;inward/=np.linalg.norm(inward);side=np.array([-inward[1],inward[0],0.])
        outward=-d.geom_xmat[hid].reshape(3,3)[:,1];tangent=np.cross([0.,0.,1.],outward)
        rows.append(publish(batch,run,cfg,xml,state,'O',None,task_index))
        chosen={}
        for category in ['B','H']:
            candidates=[]
            # B: side offsets from base, short column. H: side of handle, tall column.
            for idx,(sign,offset,depth) in enumerate(itertools.product([1.,-1.],[.50,.65,.80] if category=='B' else [.25,.40,.55],[.15,.30])):
                if category=='B':pos=base+sign*offset*side+depth*inward;size=[.06,.06,.25]
                else:pos=h+sign*offset*tangent+depth*outward;size=[.045,.045,(h[2]+.12)/2]
                pos[2]=size[2]
                xm=box_xml(xml,pos,size);cm,cd=model(xm,state)
                check=obstacle_check(cm,cd,target,joint)
                row=dict(index=idx,position=pos.tolist(),half_size=size,sign=sign,offset_m=offset,depth_m=depth,**check)
                candidates.append(row)
            write(run/(prefix+'-'+category+'-candidates.json'),candidates)
            valid=[v for v in candidates if v['accepted']]
            if not valid:raise ValueError(f'No declared {prefix}/{category} obstacle passes; preserve drafts and diagnostics')
            # First passing declared candidate, independent of any task outcome.
            chosen[category]=valid[0]
            rows.append(publish(batch,run,cfg,xml,state,category,valid[0],task_index+2*(['B','H'].index(category)+1)))
        # M changes initial opening only relative to H, using exactly the same obstacle/start.
        d.qpos[m.jnt_qposadr[joint]]*=.8;mujoco.mj_forward(m,d)
        mstate=saved(m,d);checkm,checkd=model(box_xml(xml,chosen['H']['position'],chosen['H']['half_size']),mstate)
        assert obstacle_check(checkm,checkd,target,joint)['accepted']
        rows.append(publish(batch,run,cfg,xml,mstate,'M',chosen['H'],task_index+6))
        write(run/'scene-drafts.json',dict(at=stamp(),rows=rows,new_task_outcomes=0,independent_environment_total=2,primary_enabled=False))
    print(json.dumps(rows),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    a=parser.parse_args();prepare(a.batch,a.run)
