"""CPU-only restore/preview and bounded E/D/A interface probes, not outcomes."""
import argparse,json,os
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image
from OpenGL import GL
from human_scene_pilot import PilotReference,make_args,load,write_json,stamp,key_actions
from mobiwam.reference_formal_substep import FormalSubstepMonitor
from mobiwam.reference_prefix_safety import JointMarginMonitor,GuardedIntegration
from mobiwam.human_paired_protocol import validate_dock

def probe(ref,output):
    output.mkdir()
    rows=[]
    for route in ['E','D','A']:
        ref.restore();ref.bind();ref.route=route;ref.docked=False
        m,d=ref.model_data();arm=ref.robot.part_controllers['right'];base=ref.robot.part_controllers['base']
        original_base=d.qpos[base.qpos_index].copy();arm0=d.qpos[arm.qpos_index].copy()
        ids=[int(np.flatnonzero(m.jnt_qposadr==i)[0]) for i in arm.qpos_index]
        joint=JointMarginMonitor(arm.qpos_index,m.jnt_range[ids],[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in ids])
        guard=FormalSubstepMonitor(ref,ref.native['fixture_name']);path=output/route;path.mkdir()
        states=[ref.env.sim.get_state().flatten().copy()];actions=[];events=[];dock=None;failure=None
        total=45 if route=='D' else 20
        try:
            for i in range(total):
                if route=='D' and i==25:
                    validate_dock('D',False,d.qvel[base.qvel_index],d.qpos[arm.qpos_index],ref.pilot['common_stow_qpos'],False,False)
                    ref.docked=True;dock=d.qpos[base.qpos_index].copy();events.append(dict(step=i,event='engineering_dock',qvel=d.qvel[base.qvel_index].copy()))
                # E receives the same base key challenge, which must be blocked.
                # D additionally challenges arm/grasp inhibition before dock.
                keys={'w'} if i<5 or (route=='D' and i>=25) else set()
                if route=='D' and i<5:keys.add('up')
                parts=ref.prepare_live_action(key_actions(keys,route=='D' and i<5,False))
                action=ref.robot.create_action_vector(parts)
                phase='navigate' if route=='D' and not ref.docked else 'manipulate'
                guard.set_boundary(i,phase)
                with guard,GuardedIntegration(ref.env.sim,d,joint,lite_physics=ref.env.lite_physics,step=i,phase=phase):
                    ref.env.step(action)
                actions.append(action);states.append(ref.env.sim.get_state().flatten().copy())
        except Exception as exc:
            failure=dict(type=type(exc).__name__,detail=str(exc))
        finally:
            np.savez_compressed(path/'control.npz',states=states,actions=actions)
            write_json(path/'native.json',guard.save(path));write_json(path/'joint.json',joint.receipt())
        q=np.asarray(states)[:,1:1+m.nq];baseq=q[:,base.qpos_index]
        locked=baseq if route=='E' else baseq[26:] if route=='D' and dock is not None else None
        anchor=original_base if route=='E' else dock
        drift=float(np.max(abs(locked-anchor))) if locked is not None and len(locked) else None
        delta=float(np.linalg.norm(baseq[-1,:2]-baseq[0,:2]))
        record=dict(route=route,steps=len(actions),planned_steps=total,events=events,failure=failure,
                    base_net_xy_m=delta,locked_base_max_generalized_drift=drift,
                    success=bool(ref.env._check_success()),new_human_or_task_outcomes=0)
        record['passed']=failure is None and len(actions)==total and not record['success'] and (drift is None or drift<.005)
        if route in ('D','A'):record['passed']=record['passed'] and delta>1e-5
        rows.append(record);write_json(output/'result.json',dict(at=stamp(),rows=rows,engineering_only=True))
    return rows

def check(batch,run):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('LIBGL_ALWAYS_SOFTWARE')=='1'
    rows=load(run/'scene-drafts.json')['rows'];results=[]
    for prefix in ['MW','DR']:
        subset=[row for row in rows if row['scene_id'].startswith(prefix)]
        first=load(Path(subset[0]['config']))
        args=make_args(run/(prefix+'-environment'),first['task'],first['environment_seed'],Path(first['source']))
        args.width=960;args.height=540
        ref=PilotReference(args,first)
        try:
            for row in subset:
                cfg=load(Path(row['config']));ref.pilot=cfg;ref.source=Path(cfg['source']);ref.restore();ref.bind()
                m,d=ref.model_data();before=ref.integration().copy();rng=json.dumps(ref.env.rng.bit_generator.state,sort_keys=True)
                assert np.array_equal(before,np.load(ref.source/'integration.npy'))
                assert not ref.env._check_success()
                root=Path(row['config']).parent;preview=root/'preview'
                if preview.exists():assert not any(preview.iterdir()), 'Existing preview must not be overwritten'
                else:preview.mkdir()
                cameras=dict(main=cfg['main_camera'],panoramic=cfg['panoramic_camera'])
                for name,cam in cameras.items():
                    Image.fromarray(ref.frame(cam).copy()).save(preview/(name+'.png'))
                renderer=GL.glGetString(GL.GL_RENDERER).decode()
                assert 'llvmpipe' in renderer.lower(),renderer
                assert np.array_equal(before,ref.integration()) and rng==json.dumps(ref.env.rng.bit_generator.state,sort_keys=True)
                result=dict(scene_id=row['scene_id'],at=stamp(),restore_max_abs_error=ref.restore_receipt['max_abs_error'],
                            rng_unchanged=True,zero_actions=True,renderer=renderer,identity=ref.native,
                            previews={n:str(preview/(n+'.png')) for n in cameras},visual_review='pending',primary_enabled=False)
                write_json(preview/'receipt.json',result);results.append(result)
                if row['category']=='O':
                    result['protocol_probe']=probe(ref,run/(prefix+'-protocol-probe'))
                write_json(run/'checks.json',dict(at=stamp(),scenes=results,formal_train_ready=False))
        finally:
            for name in ['renderer','observation_renderer']:
                x=getattr(ref,name,None)
                if x is not None:x.close()
            ref.env.close()
    print('DRAFT_CHECKS_COMPLETE',len(results),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();check(a.batch,a.run)
