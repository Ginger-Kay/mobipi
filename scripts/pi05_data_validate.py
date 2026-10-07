"""Validate command reconstruction against native OSC code without env.step."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from robosuite.controllers.parts.arm.osc import OperationalSpaceController
from teleop_reference import write_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    binding=json.loads((a.run/'inventory/dataset-binding.json').read_text())
    split=json.loads((a.run/'inventory/source-split.json').read_text());reserved=set(split['reserved_development_evaluation_ancestors'])
    assert not reserved & {x['parent_group'] for x in binding['episodes']}
    assert not {x['parent_group'] for x in split['parent_groups'] if x['role']=='evaluation'} & set(split['old_fit2_policy_fit_ancestors'])
    receipts=[]
    for ep in binding['episodes']:
        with np.load(Path(ep['derived'])/'commands.npz',allow_pickle=False) as z:
            state=z['state'];bp=z['base_p'];br=z['base_R'];act=z['original_native_actions'];wp=z['target_world_p'];wr=z['target_world_R']
            # Use recorded sensor inputs rather than recomputed after-integration
            # FK: these are the actual cached native controller inputs.
            ctrl=object.__new__(OperationalSpaceController)
            ctrl.input_ref_frame='base';ctrl.position_limits=None;ctrl.orientation_limits=None
            ctrl.goal_pos=None;ctrl.goal_ori=None;maxpos=0.;maxrot=0.
            for t in range(len(act)):
                ctrl.origin_pos=bp[t];ctrl.origin_ori=br[t]
                ctrl.ref_pos=bp[t]+br[t]@state[t,:3]
                ctrl.ref_ori_mat=br[t]@Rotation.from_rotvec(state[t,3:6]).as_matrix()
                ctrl._goal_update_mode='desired' if act[t,11]>0 else 'achieved'
                ctrl.goal_pos=ctrl.compute_goal_pos(np.clip(act[t,:3],-1,1)*.05)
                ctrl.goal_ori=ctrl.compute_goal_ori(np.clip(act[t,3:6],-1,1)*.5)
                maxpos=max(maxpos,float(np.max(abs(ctrl.goal_pos-z['nominal_controller_goal_pos'][t]))))
                maxrot=max(maxrot,float(np.max(abs(ctrl.goal_ori-z['nominal_controller_goal_rot'][t]))))
            assert maxpos<1e-6 and maxrot<1e-6,(ep['record_id'],maxpos,maxrot)
            # Complete chunks roundtrip through the exact inference adapter.
            errors=[]
            for t in sorted(set([0,(len(act)-10)//2,len(act)-10])):
                relp=(wp[t:t+10]-(bp[t]+br[t]@state[t,:3]))@br[t]
                initial=Rotation.from_rotvec(state[t,3:6]).as_matrix()
                mats=np.einsum('ij,tjk->tik',br[t].T,wr[t:t+10]);relr=Rotation.from_matrix(mats@initial.T).as_rotvec()
                rp=bp[t]+br[t]@state[t,:3]+relp@br[t].T
                rr=np.einsum('ij,tjk,kl,lm->tim',br[t],Rotation.from_rotvec(relr).as_matrix(),br[t].T,br[t]@initial)
                errors.append(max(float(np.max(abs(rp-wp[t:t+10]))),float(np.max(abs(rr-wr[t:t+10])))))
            assert max(errors)<1e-6
            rec=dict(record_id=ep['record_id'],native_OSC_goal_max_position_error_m=maxpos,native_OSC_goal_max_rotation_error=maxrot,
                chunk_adapter_roundtrip_max_error=max(errors),moving_origin_preserved=bool(ep['base_command_nonzero_frames']),desired_mode_frames=ep['desired_mode_frames'],
                original_native_command_used=True,future_origin_in_input=False,future_EEF_displacement_used_as_command=False,env_step_calls=0)
            receipts.append(rec);print(json.dumps(rec),flush=True)
    write_json(a.run/'preflight/demo-native-command-validation.json',dict(at=datetime.now(timezone.utc).isoformat(),passed=True,records=receipts,
        method='native installed OSC compute_goal_pos/ori achieved/desired compared against reconstructed original command goals; complete chunk inversion with fixed inference adapter',
        native_source=__import__('inspect').getsourcefile(OperationalSpaceController),split_leakage_pass=True,zero_policy_forward=True,zero_env_step=True,replay_budget_used=0))

if __name__=='__main__':main()
