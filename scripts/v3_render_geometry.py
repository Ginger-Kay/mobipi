#!/usr/bin/env python3
"""Render source/IK geometry diagnostics without stepping or querying policy."""
import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image,ImageDraw


def main(root):
    camera_root=Path('/share/jhk/MobiWAM/artifacts/MMWAM-OBC-002/v1-source-video-v1.0.1/20260906T060000Z-v1-video-compat-v1.0.1')
    cameras=json.loads((camera_root/'camera-freeze-v1.0.json').read_text())['cameras']
    destination=root/'geometry-frames-v2';destination.mkdir(exist_ok=True)
    inventory=[]
    for task in ('CloseDrawer','CloseSingleDoor'):
        binding=json.loads((root/'probes'/task/'binding.json').read_text())
        source=Path(binding['source']['snapshot_path']);m=mujoco.MjModel.from_xml_path(str(source/'model.xml'));d=mujoco.MjData(m)
        state=np.load(source/'sim_state.npy');q=state[1:1+m.nq]
        camera=cameras[task];cid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,camera['name'])
        m.cam_pos[cid]=camera['position_world'];m.cam_quat[cid]=camera['quaternion_wxyz'];m.cam_fovy[cid]=camera['fovy_deg']
        m.vis.global_.offwidth=1920;m.vis.global_.offheight=1080
        entries=[('original',q)]
        folder=root/'pose-compiler'/task/'hard-feasibility-repair'
        entries.extend((p.stem,np.load(p)) for p in sorted(folder.glob('*-qpos.npy')))
        robot_ids=[i for i in range(m.ngeom) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith(('robot0','mobilebase0','gripper0'))]
        envelope=[]
        for _,pose in entries:
            d.qpos[:]=pose;mujoco.mj_forward(m,d);envelope.extend(d.geom_xpos[robot_ids].copy())
        points=np.asarray(envelope);low=points.min(0)-.35;high=points.max(0)+.35
        center=(low+high)/2;span=high-low
        height=max(1.6,span[1]/(2*np.tan(np.pi/6)),span[0]/(2*np.tan(np.pi/6)*16/9))
        m.cam_pos[cid]=[center[0],center[1],high[2]+height];m.cam_quat[cid]=[1,0,0,0];m.cam_fovy[cid]=60
        (destination/f'{task}-camera.json').write_text(json.dumps({'position':m.cam_pos[cid].tolist(),'quaternion':[1,0,0,0],
             'fovy':60,'geometry_envelope_lower':low.tolist(),'geometry_envelope_upper':high.tolist(),
             'lineage':'static geometry rerender; parent V1 camera cropped these V3 sources; no env.step or policy query'},indent=2)+'\n')
        renderer=mujoco.Renderer(m,height=1080,width=1920)
        frames=[]
        for label,pose in entries:
            d.qpos[:]=pose;mujoco.mj_forward(m,d);renderer.update_scene(d,camera=camera['name'])
            frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,1920,60),fill='black')
            draw.text((20,20),f'{task} {label} - GEOMETRY DIAGNOSTIC - zero env.step / zero policy queries',fill='white')
            path=destination/f'{task}-{label}.png';frame.save(path)
            frames.append(frame.resize((640,360)))
            inventory.append({'task':task,'label':label,'path':str(path),'width':1920,'height':1080,'outcome':False})
        sheet=Image.new('RGB',(1920,720))
        for i,frame in enumerate(frames):sheet.paste(frame,((i%3)*640,(i//3)*360))
        sheet.save(destination/f'{task}-contact-sheet.jpg')
        renderer.close()
        print(task,len(frames),'static diagnostic frames',flush=True)
    (destination/'inventory.json').write_text(json.dumps({'pid':os.getpid(),'env_step_calls':0,'policy_queries':0,'frames':inventory},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
