import mujoco
import pytest
from mobiwam.visible_object_binding import geometry_inventory,model_geometry_sha,validate_native_geometry,frame_binding,VisibleBindingHold

def native():
 m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint type="slide"/><geom type="box" size=".1 .2 .3"/></body></worldbody></mujoco>')
 return m,mujoco.MjData(m)

def test_two_cameras_keep_fresh_state_and_same_geometry():
 m,d=native();baseline=geometry_inventory(m);digest=model_geometry_sha(m)
 validate_native_geometry(m,baseline,geometry_sha=digest)
 for step in range(3):
  d.time=step*.05;d.qpos[0]=step*.01
  for camera in ('target','panorama'):
   assert frame_binding(m,d,step,camera,geometry_sha=digest)==frame_binding(m,d,step,camera)

def test_fresh_pre_and_post_action_digests_reject_model_edits():
 m,d=native();baseline=geometry_inventory(m)
 for field in ('geom_pos','geom_contype','body_pos'):
  array=getattr(m,field);before=array.copy();array.flat[0]+=1
  for provided in (False,True):
   kwargs={'geometry_sha':model_geometry_sha(m)} if provided else {}
   with pytest.raises(VisibleBindingHold,match='binding changed'):validate_native_geometry(m,baseline,**kwargs)
  array[:]=before
  assert validate_native_geometry(m,baseline)
