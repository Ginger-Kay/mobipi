"""Bind one original native snapshot per frozen parent; zero task actions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import traceback
import mujoco
import numpy as np
from PIL import Image
from teleop_reference import Reference, write_json
from human_scene_pilot import restore_saved_integration
from mobiwam.adapters.mobipi import _capture_controller_state
from mobiwam.task_video_identity import observe_native
from mobiwam.pi05_adapter import observation, CAMERAS
from mobiwam.pi05_motion import arm_indices
from mobiwam.reference_collision import SweptGeometry


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    a = ap.parse_args()
    split = json.loads((a.run/'inventory/source-split.json').read_text())
    records = []
    for g in split['parent_groups']:
        co = a.run/'design/anchors'/g['config_id']
        co.mkdir(parents=True, exist_ok=True)
        original = Path(g['source'])
        dest = co/original.name
        dest.mkdir(exist_ok=True)
        for name in ('model.xml','integration.npy','ep_meta.json','rng.json','source.json','target-binding.json'):
            shutil.copy2(original/name, dest/name)
        shutil.copy2(original.parent/'env_config.json', co/'env_config.json')
        ref = None
        row = dict(g, source=str(dest), original_source=str(original), slot=1, tier=1,
                   anchor_selection='original demonstration Source snapshot; no neutral reset or base perturbation',
                   contact_initialized=None, known_development=True, template_status='seen-template',
                   status='pending', env_step_calls=0, policy_forward_calls=0)
        try:
            ref = Reference(argparse.Namespace(output=str(co/'native'), task=g['task'], layout=1, style=0,
                seed=g['environment_seed'], self_test=True, source=str(dest), replay_attempt=None,
                resume_attempt=None, width=640, height=360))
            restore_saved_integration(ref)
            def no_step(*args, **kwargs):
                raise AssertionError('anchor binding must not advance physics')
            ref.env.step = no_step
            m,d = ref.model_data()
            saved = np.load(dest/'integration.npy')
            scratch = mujoco.MjData(m)
            mujoco.mj_setState(m, scratch, saved, ref.kind)
            poserr = float(np.max(abs(d.qpos-scratch.qpos)))
            velerr = float(np.max(abs(d.qvel-scratch.qvel)))
            assert max(poserr,velerr) <= 1e-6
            target = json.loads((dest/'target-binding.json').read_text())
            ref.identity_expected = dict(task=g['task'], fixture_name=target['fixture_name'],
                fixture_class=target['fixture_class'], model_sha256=sha(dest/'model.xml'))
            native = observe_native(ref, ref.identity_expected)
            assert native['task'] == g['task']
            fixture = ref.env.drawer if g['task']=='CloseDrawer' else ref.env.door_fxtr
            qids,_,limits = arm_indices(ref)
            margin = float(np.min(np.minimum(d.qpos[qids]-limits[:,0],limits[:,1]-d.qpos[qids])))
            check = SweptGeometry(m,target_prefix=fixture.name,margin=.0005)
            _, distances = check.distances(d.qpos,'manipulate')
            clearance = float(np.min(distances,initial=.1))
            contact_names = []
            for c in d.contact[:d.ncon]:
                names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(i)) or '' for i in [c.geom1,c.geom2]]
                if any(fixture.name in n for n in names) and any('gripper' in n or 'finger' in n for n in names):
                    contact_names.append(names)
            inputs,_ = observation(ref)
            np.savez_compressed(co/'policy-input.npz', **inputs)
            for key in CAMERAS:
                Image.fromarray(inputs[key]).save(co/(key+'.png'))
            controller = _capture_controller_state(ref.env)
            qpos = d.qpos.copy(); qvel = d.qvel.copy()
            restore_saved_integration(ref)
            m,d = ref.model_data()
            assert np.max(abs(d.qpos-qpos)) <= 1e-6 and np.max(abs(d.qvel-qvel)) <= 1e-6
            reasons = []
            if margin <= .015: reasons.append('original initial arm joint margin <= .015rad')
            if clearance < .0005: reasons.append('original initial forbidden clearance < .5mm')
            if ref.env._check_success(): reasons.append('original snapshot already checker success')
            row.update(status='X' if reasons else 'bound', rejection_reasons=reasons,
                native=native,contact_initialized=bool(contact_names),initial_target_contacts=contact_names,
                joint_margin_rad=margin,initial_forbidden_clearance_m=clearance,
                qpos_max_abs_error=poserr,qvel_max_abs_error=velerr,native_model_dimensions=dict(nq=m.nq,nv=m.nv,nu=m.nu),
                policy_input=str(co/'policy-input.npz'),policy_camera_mapping=CAMERAS,
                source_sha256={n:sha(dest/n) for n in ('model.xml','integration.npy','rng.json','target-binding.json')},
                controller_restore='native Reference.restore refresh; original controller history not separately serialized',
                controller_history_qualification='native refresh verified; original history unavailable',
                restore=ref.restore_receipt)
            write_json(co/'initial-state-controller.json',dict(at=now(),qpos=qpos.tolist(),qvel=qvel.tolist(),controller=controller,
                original_snapshot_preserved=True,restore=ref.restore_receipt,env_step_calls=0))
        except Exception:
            row.update(status='X',rejection_reasons=['native source identity/restore binding failed'],traceback=traceback.format_exc())
        finally:
            if ref is not None:
                if getattr(ref,'pi05_renderer',None): ref.pi05_renderer.close()
                ref.env.close()
        write_json(co/'binding.json',dict(at=now(),**row))
        records.append(row)
        print(json.dumps(dict(at=now(),config_id=g['config_id'],status=row['status'],bound=len(records))),flush=True)
    write_json(a.run/'design/start-design.json',dict(at=now(),parent_count=20,config_count=20,selected=records,
        zero_policy_forward=True,zero_env_step=True,one_anchor_per_parent=True,near_neighbor_search_used=False,
        X=[r['config_id'] for r in records if r['status']=='X']))
    dev = [min((r for r in records if r['role']=='development' and r['task']==task),key=lambda r:r['config_id'])
        for task in ('CloseDrawer','CloseSingleDoor')]
    slots = [dict(r,route=route) for r in dev for route in 'EDA']
    write_json(a.run/'policy/policy-dev-roster.json',dict(at=now(),selection='stable config_id minimum per task; no policy outcome used',
        parents=[r['parent_group'] for r in dev],slots=slots,routes=['E','D','A'],policy_sampling_seed=20261008,
        simulation_seconds=120,wall_seconds=2700,max_development_science=30))


if __name__ == '__main__':
    main()
