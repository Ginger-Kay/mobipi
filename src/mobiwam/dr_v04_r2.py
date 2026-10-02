"""R2 prospective binding and deterministic zero-action evidence-camera selection."""
import json,math
from pathlib import Path
import numpy as np

VERSION='DR-v0.4-R2-batched-train-validation'
CAMERA_RULE='target-base-side-segmentation-depth-v3'
ADMITTED='CloseDrawer-layout1-style0-seed109'
SAFETY=dict(collision_margin_m=.0005,joint_margin_strict_rad=.015,replay_max_state_abs_error=1e-5)

def validate_binding(freeze,binding,group_id,execution_sha):
    order=freeze['task_interleaved_train_validation_order']
    if order[0]!=ADMITTED or len(order)!=36 or len(set(order))!=36:raise ValueError('parent schedule differs')
    allowed=order[1:]
    if binding.get('version')!=VERSION or binding.get('execution_code_commit')!=execution_sha:raise ValueError('R2 execution code differs')
    if binding.get('parent_formal_execution_code_commit')!=freeze['formal_execution_code_commit']:raise ValueError('R2 parent code differs')
    if binding.get('allowed_group_ids')!=allowed or group_id not in allowed:raise ValueError('R2 sealed/seen/unregistered Source')
    expected=[next(x for x in freeze['primary'] if x['group_id']==g) for g in allowed]
    if binding.get('scientific_rows')!=expected:raise ValueError('R2 scientific Source/plan/split/order changed')
    if any(x['split'] not in ('train','validation') for x in expected):raise ValueError('R2 test isolated')
    if binding.get('safety')!=SAFETY:raise ValueError('R2 safety thresholds changed')
    if binding.get('camera_rule')!=CAMERA_RULE:raise ValueError('unfrozen camera rule')
    if not binding.get('run_id') or binding.get('new_route_budget')!=105:raise ValueError('R2 run identity/budget missing')
    if binding.get('training_authorized') is not False:raise ValueError('R2 training forbidden')
    return next(x for x in expected if x['group_id']==group_id)

def camera_candidates(handle,base):
    handle=np.asarray(handle,float);base=np.asarray(base,float)
    approach=math.degrees(math.atan2(base[1]-handle[1],base[0]-handle[0]))
    center=.60*handle+.40*base;center[2]=max(.65,float(handle[2])*.65)
    distance=max(2.0,float(np.linalg.norm(handle-base))*1.8)
    return [dict(lookat=center.tolist(),distance=distance,azimuth=(approach+180+offset)%360,elevation=-18.)
            for offset in (0,30,-30,60,-60,90,-90,120,-120,150,-150,180)]

def prepare_camera(ref,row,output):
    import mujoco
    from PIL import Image,ImageDraw
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    before=ref.integration().copy();rng=json.dumps(ref.env.rng.bit_generator.state,sort_keys=True)
    m,d=ref.model_data()
    bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,ref.base_body)
    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(m.ngeom)]
    sets=dict(target=[i for i,n in enumerate(names) if n.startswith(row['fixture_name'])],
              handle=[i for i,n in enumerate(names) if n.startswith(row['fixture_name']) and 'handle' in n],
              gripper=[i for i,n in enumerate(names) if n.startswith('gripper0_')],
              base=[i for i,n in enumerate(names) if n.startswith('mobilebase0_')])
    if any(not x for x in sets.values()):raise ValueError('native visibility geometry missing')
    candidates=camera_candidates(np.mean(d.geom_xpos[sets['handle']],axis=0),d.xpos[bid]);records=[];frames=[]
    for i,c in enumerate(candidates):
        rgb=ref.frame(c).copy();frames.append(rgb)
        ref.renderer.enable_segmentation_rendering()
        try:seg=ref.frame(c).copy()
        finally:ref.renderer.disable_segmentation_rendering()
        valid=seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM)
        counts={k:int(np.count_nonzero(valid & np.isin(seg[:,:,0],ids))) for k,ids in sets.items()}
        # Native handle collision geoms are intentionally hidden by the normal
        # recorder's render options. Measure the visible target surface at the
        # projected handle location using depth, without changing geom groups,
        # collision masks, renderer policy, or physical geometry.
        ref.renderer.enable_depth_rendering()
        try:depth=ref.frame(c).copy()
        finally:ref.renderer.disable_depth_rendering()
        cam=ref.renderer.scene.camera[0]
        forward=np.asarray(cam.forward);up=np.asarray(cam.up)
        right=np.cross(forward,up);delta=np.mean(d.geom_xpos[sets['handle']],axis=0)-np.asarray(cam.pos)
        z=float(np.dot(delta,forward));focal=ref.args.height/(2*math.tan(math.radians(float(m.vis.global_.fovy))/2))
        handle_pixels=0
        if z>0:
            px=ref.args.width/2+focal*float(np.dot(delta,right))/z
            py=ref.args.height/2-focal*float(np.dot(delta,up))/z
            radius=max(6,int(math.ceil(focal*.04/z)))
            yy,xx=np.ogrid[:depth.shape[0],:depth.shape[1]]
            region=(xx-px)**2+(yy-py)**2<=radius**2
            handle_pixels=int(np.count_nonzero(region & valid & np.isin(seg[:,:,0],sets['target']) & (np.abs(depth-z)<.08)))
        counts['handle']=handle_pixels
        thresholds=dict(target=800,handle=30,gripper=100,base=500)
        ratios=[counts[k]/v for k,v in thresholds.items()]
        records.append(dict(index=i,camera=c,pixels=counts,minimum_visibility_ratio=min(ratios)))
    # Outcome-blind lexicographic visual coverage: all required entities first,
    # then target and contact detail, stable candidate index breaks exact ties.
    selected=max(records,key=lambda x:(min(x['minimum_visibility_ratio'],2.),x['pixels']['handle'],x['pixels']['gripper'],-x['index']))
    Image.fromarray(frames[selected['index']]).save(output/'selected.jpg')
    panel=Image.new('RGB',(1440,300*math.ceil(len(frames)/3)))
    for i,frame in enumerate(frames):
        thumb=Image.fromarray(frame).resize((480,270));panel.paste(thumb,((i%3)*480,(i//3)*300+30))
    draw=ImageDraw.Draw(panel)
    for i,r in enumerate(records):draw.text(((i%3)*480+5,(i//3)*300+5),str(i)+' '+str(r['pixels']),fill='white')
    panel.save(output/'candidates.jpg')
    if not np.array_equal(before,ref.integration()) or rng!=json.dumps(ref.env.rng.bit_generator.state,sort_keys=True):raise ValueError('preview changed physical state/RNG')
    receipt=dict(rule=CAMERA_RULE,group_id=row['group_id'],camera=selected['camera'],selected_index=selected['index'],candidates=records,
                 zero_task_actions=True,source_integration_unchanged=True,rng_unchanged=True,policy_cameras_unchanged=ref.policy_cameras,
                 native_fovy_degrees=float(m.vis.global_.fovy),visibility_pass=selected['minimum_visibility_ratio']>=1.,
                 scope='source-only free camera segmentation; all source target/handle/gripper/base visible; full-route review remains pending')
    (output/'camera.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if not receipt['visibility_pass']:raise ValueError('camera Source visibility insufficient; geometry-only adjustment needed')
    return receipt
