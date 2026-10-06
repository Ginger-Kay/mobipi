"""Zero-action evidence camera planning; preserves policy cameras and inputs."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import os
import mujoco
import numpy as np
from PIL import Image,ImageDraw


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    rows=json.loads((a.run/'data/policy-fit-candidates.json').read_text());out=a.run/'preflight/evidence-views';out.mkdir(exist_ok=False)
    for task in ('CloseDrawer','CloseSingleDoor'):
        row=next(x for x in rows if x['task']==task);src=Path(row['attempt']).parents[1]
        xml=(src/'model.xml').read_text().replace('/share/personal/haokaijiang/MobiWAM','/share/personal/chensiyu/haokaijiang/MobiWAM')
        m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_setState(m,d,np.load(src/'integration.npy'),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
        sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'gripper0_right_grip_site');lookat=d.site_xpos[sid].copy()
        renderer=mujoco.Renderer(m,height=360,width=640);opt=mujoco.MjvOption();opt.geomgroup[0]=0
        sheet=Image.new('RGB',(4*640,2*390));draw=ImageDraw.Draw(sheet);records=[]
        for i,angle in enumerate(range(0,360,45)):
            camera=mujoco.MjvCamera();camera.lookat[:]=lookat;camera.distance=1.55;camera.azimuth=angle;camera.elevation=-12
            renderer.update_scene(d,camera=camera,scene_option=opt);im=Image.fromarray(renderer.render().copy());im.save(out/f'{task}-{angle}.png')
            sheet.paste(im,(i%4*640,i//4*390));draw.text((i%4*640+10,i//4*390+365),f'{task} azimuth={angle}',fill='white')
            records.append(dict(lookat=lookat.tolist(),distance=1.55,azimuth=angle,elevation=-12,preview=str(out/f'{task}-{angle}.png')))
        sheet.save(out/f'{task}-contact-sheet.jpg');renderer.close()
        (out/f'{task}-cameras.json').write_text(json.dumps(dict(created_at=datetime.now(timezone.utc).isoformat(),source=str(src),env_steps=0,policy_cameras_unchanged=True,cameras=records),indent=2)+'\n')
    print('ZERO ACTION PREVIEWS COMPLETE',flush=True)


if __name__=='__main__':main()
