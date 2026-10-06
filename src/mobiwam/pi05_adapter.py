"""Explicit robot observation and query-base nominal EEF action adapter."""
from io import BytesIO
import http.client
import json
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import mujoco
from reference_executor import mapped_action

CAMERAS={'base_0_rgb':'robot0_agentview_left','left_wrist_0_rgb':'robot0_eye_in_hand','right_wrist_0_rgb':'robot0_agentview_right'}


def observation(ref):
    raw=ref.env._get_observations(force_update=True);m,d=ref.model_data()
    p=np.asarray(raw['robot0_base_pos']);R=Rotation.from_quat(raw['robot0_base_quat']).as_matrix()
    ep=np.asarray(raw['robot0_eef_pos']);er=Rotation.from_quat(raw['robot0_eef_quat_site']).as_matrix()
    state=np.zeros(32,np.float32);state[:3]=R.T@(ep-p);state[3:6]=Rotation.from_matrix(R.T@er).as_rotvec()
    state[6]=np.mean(abs(raw['robot0_gripper_qpos']))/.04
    state[7:14]=raw['robot0_joint_pos_cos'];state[14:21]=raw['robot0_joint_pos_sin'];state[21]=d.qpos[3]
    if getattr(ref,'pi05_renderer_model',None)!=id(m):
        if getattr(ref,'pi05_renderer',None):ref.pi05_renderer.close()
        ref.pi05_renderer=mujoco.Renderer(m,height=256,width=256);ref.pi05_renderer_model=id(m)
    images={}
    for slot,camera in CAMERAS.items():
        ref.pi05_renderer.update_scene(d,camera=camera,scene_option=ref.render_options)
        images[slot]=np.asarray(Image.fromarray(ref.pi05_renderer.render().copy()).resize((224,224),Image.Resampling.BILINEAR))
    prompt='Close the drawer.' if ref.args.task=='CloseDrawer' else 'Close the microwave door.'
    return dict(state=state,prompt=prompt,**images),dict(base_world_p=p.copy(),base_world_R=R.copy(),sim_time=float(d.time))


def call(port,path,payload):
    conn=http.client.HTTPConnection('127.0.0.1',port,timeout=180)
    try:
        conn.request('POST',path,payload,headers={'Content-Type':'application/octet-stream'})
        response=conn.getresponse();data=response.read()
        if response.status!=200:raise RuntimeError(f'pi05 service HTTP{response.status}: '+data.decode(errors='replace'))
        return data
    finally:conn.close()


def query(port,inputs):
    buf=BytesIO();np.savez(buf,**inputs);data=call(port,'/infer',buf.getvalue())
    with np.load(BytesIO(data),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def reset(port):call(port,'/reset',json.dumps({'policy_sampling_seed':20261006}).encode())


def world_intent(action,anchor):
    action=np.asarray(action)
    if action.shape!=(32,) or not np.isfinite(action).all():raise ValueError('pi05 padded action must be finite32')
    return dict(pos=anchor['base_world_p']+anchor['base_world_R']@action[:3],
        rot=anchor['base_world_R']@Rotation.from_rotvec(action[3:6]).as_matrix(),grasp=float(np.clip(action[6],-1,1)))


def execute_static(ref,action,anchor,base_target):
    point=world_intent(action,anchor)
    # Native achieved-goal OSC, inherited translation/rotation limits and base
    # servo. Torso and all padded channels are neutral; policy never drives base.
    actual=mapped_action(ref,point,base_target,arm_enabled=True)
    actual[10]=0.;actual[11]=-1.
    return actual,point
