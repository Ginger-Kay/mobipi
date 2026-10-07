"""Repair stale native-model snapshot aliases with the already frozen qpos.

No candidate, Source split, coordinate, physics, or outcome selection changes.
Original invalid snapshots remain available. Slot1 is copied byte-exactly.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import mujoco
import numpy as np
from teleop_reference import Reference,write_json
from human_scene_pilot import restore_saved_integration
from mobiwam.adapters.mobipi import _capture_controller_state

def now():return datetime.now(timezone.utc).isoformat()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();root=a.run/'design'
    initial=root/'initial-invalid-start-design.json'
    if not initial.exists():shutil.copy2(root/'start-design.json',initial)
    old=json.loads(initial.read_text());new=[];receipts=[]
    for g in old['selected']:
        previous=Path(g['source']);co=root/'repaired-starts-v1/configs'/g['config_id'];co.mkdir(parents=True,exist_ok=True);dest=co/previous.name
        if (co/'repair-receipt.json').exists():
            receipt=json.loads((co/'repair-receipt.json').read_text());new.append(dict(g,source=str(dest),original_source=str(previous),state_alias_repaired=g['slot']!=1));receipts.append(receipt);continue
        dest.mkdir()
        for name in ('model.xml','ep_meta.json','rng.json','target-binding.json','source.json','integration.npy'):shutil.copy2(previous/name,dest/name)
        shutil.copy2(previous.parent/'env_config.json',co/'env_config.json')
        intended=json.loads((previous.parent/'initial-state-controller.json').read_text())
        ref=Reference(argparse.Namespace(output=str(co/'native'),task=g['task'],layout=1,style=0,seed=g['environment_seed'],self_test=True,source=str(dest),replay_attempt=None,resume_attempt=None,width=640,height=360))
        try:
            restore_saved_integration(ref);m,d=ref.model_data()
            if g['slot']!=1:
                d.qpos[:]=np.asarray(intended['qpos']);d.qvel[:]=np.asarray(intended['qvel']);d.qacc_warmstart[:]=0.
                mujoco.mj_forward(m,d)
                state=np.empty(mujoco.mj_stateSize(m,ref.kind));mujoco.mj_getState(m,d,state,ref.kind)
                np.save(dest/'integration.npy',state)
                restore_saved_integration(ref)
            # Always refresh pointers after reset_from_xml_string/restore.
            m,d=ref.model_data();poserr=float(np.max(abs(d.qpos-np.asarray(intended['qpos']))));velerr=float(np.max(abs(d.qvel-np.asarray(intended['qvel']))))
            assert poserr<=1e-6 and velerr<=1e-6,(g['config_id'],poserr,velerr)
            info=json.loads((dest/'source.json').read_text());info.update(trace=ref.trace(),trace_scope='this current posed config',metadata_corrected_at=now(),parent_anchor_trace_receipt=str(previous/'source.json'))
            write_json(dest/'source.json',info)
            receipt=dict(at=now(),issue='START-MODEL-REFERENCE-001',config_id=g['config_id'],slot=g['slot'],original_source=str(previous),repaired_source=str(dest),
                frozen_intended_state_receipt=str(previous.parent/'initial-state-controller.json'),qpos_max_abs_error=poserr,qvel_max_abs_error=velerr,
                slot1_original_integration_byte_exact=(dest/'integration.npy').read_bytes()==(previous/'integration.npy').read_bytes() if g['slot']==1 else None,
                new_generation_parameters=False,new_candidate_selection=False,new_policy_outcomes_used=False,env_step_calls=0,
                controller=_capture_controller_state(ref.env),restore=ref.restore_receipt)
            write_json(co/'repair-receipt.json',receipt);write_json(co/'initial-state-controller.json',dict(intended,controller=receipt['controller'],repair_receipt=str(co/'repair-receipt.json')))
            new.append(dict(g,source=str(dest),original_source=str(previous),state_alias_repaired=g['slot']!=1));receipts.append(receipt)
            print(json.dumps(dict(at=now(),config_id=g['config_id'],qpos_error=poserr,configs=len(new))),flush=True)
        finally:ref.env.close()
    for parent in {g['parent_group'] for g in new}:
        configs=[g for g in new if g['parent_group']==parent];states=[]
        for g in configs:
            q=np.asarray(json.loads((Path(g['source']).parent/'initial-state-controller.json').read_text())['qpos'])
            assert not any(np.max(abs(q-p))<1e-6 for p in states);states.append(q)
        hashes=[(Path(g['source'])/'integration.npy').read_bytes() for g in configs];assert len(set(hashes))==len(hashes)
    repaired=dict(old,at=now(),selected=new,state_interface_repair='native stale model pointers; exact frozen intended qpos/qvel, no candidate parameters changed',
        previous_invalid_design=str(initial),state_distinctness_verified=True,slot1_dev_states_unchanged=True,zero_policy_forward=True,zero_env_step=True)
    write_json(root/'repaired-starts-v1/repair-summary.json',dict(at=now(),issue='START-MODEL-REFERENCE-001',passed=True,configs=len(new),records=receipts,main_science_executed=0))
    write_json(root/'start-design.json',repaired)

if __name__=='__main__':main()
