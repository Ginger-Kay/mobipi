"""Verify new originals and index all views; state-render supplements are diagnostic."""
import argparse,csv,hashlib,html,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image
from mobiwam.task_video_identity import source_model

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def csvwrite(p,rows):
    fields=list(dict.fromkeys(k for x in rows for k in x))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def diagnostic(receipt,out):
    """Render saved qpos only. No env.step, inference, or episode continuation."""
    attempt=Path(receipt['attempt']);source=attempt.parents[1]
    model=source_model(str(source/'model.xml'),sha(source/'model.xml'));data=mujoco.MjData(model)
    z=np.load(attempt/'formal-native-substeps.npz',allow_pickle=False)
    data.qpos[:]=z['qpos'][0];mujoco.mj_forward(model,data)
    binding=load(source/'target-binding.json');fixture=binding['fixture_name']
    # The task target and gripper determine an evidence-camera focus only.
    target_name=fixture+('_door_handle_main' if receipt['task']=='CloseDrawer' else '_door')
    target=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,target_name)
    if target<0:raise ValueError('missing actual native target body')
    tool=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'gripper0_right_hand')
    target_pos=data.xpos[target].copy();robot_pos=data.xpos[tool].copy() if tool>=0 else target_pos.copy()
    focus=(target_pos+robot_pos)/2
    model.vis.global_.offwidth=max(model.vis.global_.offwidth,1280);model.vis.global_.offheight=max(model.vis.global_.offheight,720)
    renderer=mujoco.Renderer(model,height=720,width=1280);option=mujoco.MjvOption();option.geomgroup[0]=0
    cam=mujoco.MjvCamera();cam.lookat[:]=focus;cam.distance=max(1.25,float(np.linalg.norm(robot_pos-target_pos))*2+1.)
    cam.elevation=-8.
    out.mkdir(parents=True,exist_ok=False);previews=[]
    try:
        for az in (0,90,180,270):
            cam.azimuth=az;renderer.update_scene(data,camera=cam,scene_option=option);p=out/f'preview-{az}.jpg'
            Image.fromarray(renderer.render()).save(p);previews.append(str(p))
        write(out/'camera-options.json',dict(target_body=target_name,lookat=focus.tolist(),distance=cam.distance,elevation=cam.elevation,
            previews=previews,status='select a readable native target side before rendering diagnostic video',diagnostic_only=True,
            original_model_sha256=sha(source/'model.xml'),native_state_sha256=sha(attempt/'formal-native-substeps.npz')))
    finally:renderer.close()

def render_diagnostic(receipt,out,azimuth):
    attempt=Path(receipt['attempt']);source=attempt.parents[1];cfg=load(out/'camera-options.json')
    model=source_model(str(source/'model.xml'),cfg['original_model_sha256']);data=mujoco.MjData(model)
    model.vis.global_.offwidth=max(model.vis.global_.offwidth,1280);model.vis.global_.offheight=max(model.vis.global_.offheight,720)
    z=np.load(attempt/'formal-native-substeps.npz',allow_pickle=False)
    cam=mujoco.MjvCamera();cam.lookat[:]=cfg['lookat'];cam.distance=cfg['distance'];cam.elevation=cfg['elevation'];cam.azimuth=azimuth
    renderer=mujoco.Renderer(model,height=720,width=1280);option=mujoco.MjvOption();option.geomgroup[0]=0
    frames=[]
    for control in range(receipt['steps']):
        where=np.flatnonzero(z['step_index']==control)
        if not len(where):raise ValueError('missing native control interval')
        frames.append(int(where[-1])+1)
    if len(z['step_index']) and z['step_index'][-1]>=receipt['steps']:frames.append(len(z['qpos'])-1)
    raw=out/'saved-native-state-render.mp4';writer=imageio.get_writer(raw,fps=20,codec='libx264',quality=8,macro_block_size=1)
    try:
        for i in frames:
            data.qpos[:]=z['qpos'][i];data.time=float(z['sim_time'][i]);mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera=cam,scene_option=option);writer.append_data(renderer.render())
    finally:writer.close();renderer.close()
    labeled=out/'DIAGNOSTIC-saved-native-states.mp4'
    filt='drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text=DIAGNOSTIC saved native states - BC E failure - no new execution:x=12:y=12:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.7'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(raw),'-vf',filt,'-c:v','libx264','-crf','18','-an',str(labeled)],check=True)
    write(out/'diagnostic-binding.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_attempt=str(attempt),
        original_video=str(attempt/'original.mp4'),original_video_sha256=sha(attempt/'original.mp4'),native_state_sha256=cfg['native_state_sha256'],
        diagnostic_video=str(labeled),diagnostic_sha256=sha(labeled),frames=len(frames),azimuth=azimuth,
        new_execution=False,independent_recorded_camera=False,state_re_render=True,includes_native_stop_tail=len(frames)>receipt['steps'],
        policy_queries=0,environment_steps=0,scope='readability supplement only; originals retained; not a new rollout or human acceptance'))
    print('diagnostic',labeled,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--stage',choices=['index','preview','render'],default='index')
    ap.add_argument('--slot',type=int);ap.add_argument('--azimuth',type=float);ap.add_argument('--index-version',type=int,default=1);a=ap.parse_args();r=a.run
    receipts=[load(p) for p in sorted((r/'policy').glob('ability-*/completed.json'))]
    if a.stage!='index':
        x=next(x for x in receipts if x['slot']==a.slot);out=r/'videos'/f"diagnostic-slot-{a.slot}"
        if a.stage=='preview':diagnostic(x,out)
        else:render_diagnostic(x,out,a.azimuth)
        return
    rows=[]
    for x in load(r/'data/inventory.json'):
        p=x.get('video_path','')
        if not p:continue
        rows.append(dict(record_id=x['record_id'],view=x['view'],task=x.get('task'),route=x['route'],controller=x['controller'],
            result=x['stop_reason'],original_path=p,sha256=x.get('video_sha256',''),exists=Path(p).exists(),
            validation='inherited integrity/checksum receipt reused',new_execution=False,human_intervention=x['human_or_auto']=='human',state_reset='initial only',diagnostic=False))
    actual=receipts+[load(p) for p in sorted((r/'episodes').glob('*/*/completed.json'))]
    for x in actual:
        attempt=Path(x['attempt']);base=dict(record_id=attempt.name,view='C_policy_auto' if 'slot' in x else 'reference_online',task=x['task'],route=x['route'],
            controller=x['controller'],result=x.get('status',x.get('result',{}).get('reason')),new_execution=True,human_intervention=False,state_reset='initial only',diagnostic=False)
        for name in ('original.mp4','panoramic.mp4'):
            p=attempt/name
            if not p.exists():continue
            if x.get('steps',x.get('result',{}).get('steps',0))==0:
                rows.append(dict(**base,camera=name[:-4],original_path=str(p),sha256=sha(p),exists=True,
                    validation='zero-full-frame native stop; movie not decodable; see partial state and native guard',frames=0));continue
            subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(p),'-f','null','-'],check=True)
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames,r_frame_rate,width,height','-of','json',str(p)],text=True))
            rows.append(dict(**base,camera=name[:-4],original_path=str(p),sha256=sha(p),exists=True,validation='full decode pass',frames=int(info['streams'][0]['nb_read_frames']),
                width=info['streams'][0]['width'],height=info['streams'][0]['height'],fps=info['streams'][0]['r_frame_rate']))
    for p in sorted((r/'videos').glob('diagnostic-*/diagnostic-binding.json')):
        x=load(p);rows.append(dict(record_id=p.parent.name,view='diagnostic',task='see source binding',route='E',controller='saved-state renderer',result='BC failure supplement',
            original_path=x['diagnostic_video'],sha256=x['diagnostic_sha256'],exists=True,validation='labeled diagnostic',new_execution=False,human_intervention=False,state_reset='saved-state rendering',diagnostic=True))
    index_out=r/'videos'/f'index-v{a.index_version}';index_out.mkdir(exist_ok=False)
    csvwrite(index_out/'videos.csv',rows);write(index_out/'videos.json',rows)
    cells=['<!doctype html><meta charset=utf-8><title>Simulation sprint videos</title><h1>Simulation sprint video index</h1><p>Human, reference, policy, and diagnostic views retain distinct evidence levels.</p>']
    for x in rows:
        cells.append('<p>'+html.escape(f"{x['view']} | {x['task']} | {x['route']} | {x['controller']} | {x['result']}")+'<br><a href="'+html.escape(x['original_path'],quote=True)+'">'+html.escape(x['original_path'])+'</a></p>')
    (index_out/'index.html').write_text('\n'.join(cells))
    for name in ('index.html','videos.csv','videos.json'):
        current=r/'videos'/name
        if current.is_symlink():current.unlink()
        elif current.exists():current.rename(r/'videos'/('initial-'+name))
        current.symlink_to(index_out/name)
    print('indexed',len(rows),'video entries',str(index_out),flush=True)

if __name__=='__main__':main()
