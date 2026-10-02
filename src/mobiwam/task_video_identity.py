"""Native task and recorder identity guards; no task outcome or qualification.

Historical media without internal bindings remain legacy/unverified. A verified
manifest is an engineering identity receipt, never human task approval.
"""
from __future__ import annotations
import hashlib,inspect,json
from pathlib import Path
import cv2,h5py,mujoco,numpy as np

class IdentityHold(ValueError):
    pass

def require(condition, detail):
    if not condition:raise IdentityHold('task_identity_hold: '+detail)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def topology(model):
    h=hashlib.sha256()
    # Derived from native loaded model, including joint/body/geom mapping.
    for key in ('names','body_parentid','body_jntnum','body_pos','body_quat',
                'jnt_type','jnt_bodyid','jnt_qposadr','jnt_axis','jnt_range',
                'geom_bodyid','geom_type','geom_size','geom_pos','geom_quat',
                'geom_dataid','mesh_vert','mesh_face'):
        value=getattr(model,key);h.update(key.encode());h.update(bytes(value) if isinstance(value,bytes) else np.asarray(value).tobytes())
    return h.hexdigest()

def observe_native(ref, expected):
    """Read environment, target, checker and native model, independently of CLI."""
    for key in ('task','fixture_name','fixture_class','model_sha256'):
        require(bool(expected.get(key)), 'missing declared '+key)
    env=ref.env;task=type(env).__name__
    require(task==expected['task'],f'declared task {expected["task"]}, loaded {task}')
    require(task in ('CloseDrawer','CloseSingleDoor'),'unsupported actual task '+task)
    attr='drawer' if task=='CloseDrawer' else 'door_fxtr';fixture=getattr(env,attr,None)
    require(fixture is not None,'missing actual target '+attr)
    require(fixture.name==expected['fixture_name'],'actual target object differs')
    require(type(fixture).__name__==expected['fixture_class'],'actual fixture class differs')
    require(type(fixture).__name__==('Drawer' if task=='CloseDrawer' else 'Microwave'),'unsupported task fixture')
    require(env.behavior=='close','checker behavior differs')
    method=env._check_success
    code=inspect.getsource(method);read=inspect.getsource(fixture.get_door_state)
    require(getattr(method,'__self__',None) is env,'checker bound to another environment')
    require('self.'+attr+'.get_door_state' in code and '0.05' in code,'unknown checker target or threshold')
    suffix='_slidejoint' if task=='CloseDrawer' else '_microjoint'
    require(suffix in read and 'joint_name2id' in read,'unknown native checker joint read')
    model,data=ref.model_data();name=fixture.name+suffix
    jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
    require(jid>=0,'checker target joint missing from native model')
    adr=int(model.jnt_qposadr[jid]);require(jid==adr,'checker joint-id/qpos-address mismatch')
    require(int(model.jnt_type[jid])==int(mujoco.mjtJoint.mjJNT_SLIDE if task=='CloseDrawer' else mujoco.mjtJoint.mjJNT_HINGE),'actual target degree of freedom differs')
    source=Path(ref.source).resolve();require(sha(source/'model.xml')==expected['model_sha256'],'declared Source model differs')
    native_source=mujoco.MjModel.from_xml_path(str(source/'model.xml'))
    require(topology(model)==topology(native_source),'loaded native model topology differs from Source')
    return dict(task=task,environment_class=type(env).__module__+'.'+task,
        language=env.get_ep_meta()['lang'],fixture_name=fixture.name,fixture_class=type(fixture).__name__,
        joint_name=name,joint_id=jid,qpos_address=adr,joint_type=int(model.jnt_type[jid]),
        checker=method.__func__.__module__+'.'+method.__func__.__qualname__,checker_target=attr,
        checker_source_sha256=hashlib.sha256(code.encode()).hexdigest(),
        target_reader_source_sha256=hashlib.sha256(read.encode()).hexdigest(),
        model_topology_sha256=topology(model),source_model_sha256=sha(source/'model.xml'),
        target_description=f'{task}: {fixture.name}, joint {name}; close opening <= 0.05')

def recorder_binding(ref, path, expected, context):
    for key in ('run_id','group_id'):
        require(bool(context.get(key)),'missing recording '+key)
    path=Path(path).resolve();require(ref.route in ('E','D','A'),'missing recording route')
    native=observe_native(ref,expected)
    return dict(schema='native-task-video-identity-v1',run_id=context['run_id'],group_id=context['group_id'],
        route=ref.route,attempt_id=path.name,attempt=str(path),source=str(Path(ref.source).resolve()),
        source_model_sha256=sha(Path(ref.source)/'model.xml'),native=native,
        policy_cameras=list(ref.policy_cameras),camera=ref.camera_state())

def finalize_recording(path,binding,steps,kind='route'):
    path=Path(path).resolve();require(kind in ('route','zero_action_diagnostic'),'unknown recording kind')
    manifest=dict(binding=binding,steps=int(steps),kind=kind,files={})
    for name in ('original.mp4','demo.hdf5','trace.jsonl'):
        p=path/name;require(p.is_file(),'missing recorded '+name)
        manifest['files'][name]=dict(path=str(p),size=p.stat().st_size,sha256=sha(p))
    p=path/'task-video-manifest.json';write(p,manifest)
    verified=validate_recording(p,context={k:binding[k] for k in ('run_id','group_id','route','attempt_id')})
    write(p,verified)
    return p

def validate_recording(manifest_path,context=None):
    manifest_path=Path(manifest_path).resolve();require(manifest_path.is_file(),'missing recorder identity manifest');v=json.loads(manifest_path.read_text());b=v.get('binding',{})
    for key in ('schema','run_id','group_id','route','attempt_id','attempt','source','source_model_sha256','native','camera','policy_cameras'):
        require(bool(b.get(key)),'missing binding '+key)
    require(b['schema']=='native-task-video-identity-v1','unknown identity schema')
    for key in ('task','environment_class','fixture_name','fixture_class','joint_name','qpos_address','checker','checker_target','checker_source_sha256','model_topology_sha256','source_model_sha256','target_description'):
        require(key in b['native'] and b['native'][key] is not None,'missing native '+key)
    path=manifest_path.parent
    require(str(path)==b['attempt'] and path.name==b['attempt_id'],'wrong video/attempt location')
    if context:
        for key,value in context.items():require(b.get(key)==value,'wrong '+key)
    require(sha(Path(b['source'])/'model.xml')==b['source_model_sha256'],'Source model changed')
    for name in ('original.mp4','demo.hdf5','trace.jsonl'):
        item=v.get('files',{}).get(name,{})
        require(item.get('path')==str(path/name),'wrong file path '+name)
        require((path/name).is_file() and item.get('sha256')==sha(path/name),'wrong video/attempt or changed '+name)
    n=v.get('steps');require(isinstance(n,int) and n>=0,'missing steps')
    trace=[json.loads(x) for x in (path/'trace.jsonl').read_text().splitlines()]
    with h5py.File(path/'demo.hdf5') as f:
        g=f['data/demo_0'];require(json.loads(g.attrs.get('task_video_identity','{}'))==b,'HDF5/manifest identity differs')
        require(hashlib.sha256(g.attrs['model_file'].encode()).hexdigest()==b['source_model_sha256'],'HDF5 source-model differs')
        require(json.loads(g.attrs['env_info'])['env_name']==b['native']['task'],'HDF5 native task differs')
        require(g['actions'].shape[0]==n and g['states'].shape[0]==n+1 and len(trace)==n,'video-HDF5-trace count differs')
        model=mujoco.MjModel.from_xml_path(str(Path(b['source'])/'model.xml'))
        for i,t in enumerate(trace):
            require(t['step']==i and t['camera']==b['camera'],'trace index or camera differs')
            require(np.max(np.abs(g['states'][i+1,1:1+model.nq]-t['after']['qpos']))<=1e-10,'HDF5/trace state differs')
    frames=[];cap=cv2.VideoCapture(str(path/'original.mp4'))
    try:
        require(cap.isOpened(),'video cannot decode')
        while True:
            ok,frame=cap.read()
            if not ok:break
            frames.append(hashlib.sha256(frame.tobytes()).hexdigest())
    finally:cap.release()
    require(len(frames)==(1 if v['kind']=='zero_action_diagnostic' else n),'video frame count differs')
    if 'decoded_frames_sha256' in v:require(v['decoded_frames_sha256']==frames,'decoded video content differs')
    v['decoded_frames_sha256']=frames
    return v

def human_delivery(manifest_path,output,context=None):
    """Generate review paths only after the actual recorder binding validates."""
    v=validate_recording(manifest_path,context);b=v['binding'];p=Path(b['attempt'])/'original.mp4'
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    from PIL import Image,ImageDraw
    cap=cv2.VideoCapture(str(p));n=len(v['decoded_frames_sha256']);sheet=Image.new('RGB',(960,220));draw=ImageDraw.Draw(sheet)
    for i,k in enumerate((0,n//2,n-1)):
        cap.set(cv2.CAP_PROP_POS_FRAMES,k);ok,frame=cap.read();require(ok,'preview decode failed')
        im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im.thumbnail((320,180));sheet.paste(im,(i*320,35))
        draw.text((i*320+4,8),f'{b["route"]} frame {k}',fill='white')
    cap.release();sheet.save(out/'preview.jpg')
    review=dict(video=str(p),video_sha256=v['files']['original.mp4']['sha256'],manifest=str(Path(manifest_path).resolve()),
        target=b['native']['target_description'],preview=str((out/'preview.jpg').resolve()),
        identity_status='native_identity_verified',human_motion_semantics_review='pending',human_task_review='pending',
        machine_safety='not_assessed_by_identity_guard',formal_qualification=False,kind=v['kind'])
    write(out/'review.json',review);return review
