import copy,json
from pathlib import Path
import numpy as np
import pytest
from mobiwam.dr_v04_r2 import validate_binding,camera_candidates,VERSION,CAMERA_RULE,SAFETY
P=Path('/share/personal/chensiyu/haokaijiang/MobiWAM/artifacts/MMWAM-OBC-002-DR/DR-v0.4/20260930T115330Z-live-preflight/formal-preoutcome-freeze-v04-r1.json')

def inputs():
 f=json.loads(P.read_text());ids=f['task_interleaved_train_validation_order'][1:]
 b=dict(version=VERSION,execution_code_commit='code',parent_formal_execution_code_commit=f['formal_execution_code_commit'],allowed_group_ids=ids,scientific_rows=[next(r for r in f['primary'] if r['group_id']==g) for g in ids],safety=SAFETY.copy(),camera_rule=CAMERA_RULE,run_id='unique',new_route_budget=105,training_authorized=False)
 return f,copy.deepcopy(b),ids[0]

def test_original_order_and_rows_preserved():
 f,b,g=inputs();assert validate_binding(f,b,g,'code')['group_id']==g

@pytest.mark.parametrize('change',['seen','test','order','source','threshold','training','code'])
def test_reject_boundary_changes(change):
 f,b,g=inputs()
 if change=='seen':g=f['task_interleaved_train_validation_order'][0]
 if change=='test':g=next(r['group_id'] for r in f['primary'] if r['split']=='sealed_test')
 if change=='order':b['allowed_group_ids']=b['allowed_group_ids'][::-1]
 if change=='source':b['scientific_rows'][0]['model_sha256']='tampered'
 if change=='threshold':b['safety']['collision_margin_m']=0
 if change=='training':b['training_authorized']=True
 if change=='code':b['execution_code_commit']='other'
 with pytest.raises(ValueError):validate_binding(f,b,g,'code')

def test_camera_translation_covariance():
 h=np.array([1.,2.,1.]);b=np.array([0.,1.,.3]);delta=np.array([4.,-3.,0.])
 a=camera_candidates(h,b);c=camera_candidates(h+delta,b+delta)
 for x,y in zip(a,c):
  assert np.allclose(np.array(y['lookat'])-x['lookat'],delta)
  assert x['distance']==pytest.approx(y['distance']);assert x['azimuth']==pytest.approx(y['azimuth'])
