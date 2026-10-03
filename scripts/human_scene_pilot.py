"""Two-day human collection pilot. No automatic task actions or training.

prepare: new native scene, saved open/precontact Source, restore check and two
evidence cameras, zero task actions. ui: starts paused, operator controls actions.
record-smoke: two zero-input recorder steps, isolated engineering attempt only.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import mujoco
import numpy as np
from PIL import Image
from teleop_reference import Reference, write_json, stamp, key_actions
from mobiwam.human_scene_collection import VERSION, validate_start, append_index, load_index, phase_for
from mobiwam.task_video_identity import observe_native
from mobiwam.dr_v04_r2 import prepare_camera
from mobiwam.reference_formal_substep import FormalSubstepMonitor, FormalSafetyStop
from mobiwam.reference_prefix_safety import JointMarginMonitor, JointMarginStop, GuardedIntegration


def load(p): return json.loads(Path(p).read_text())
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def code_commit():
    return subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[1]), 'rev-parse', 'HEAD'], text=True).strip()


class PilotReference(Reference):
    def __init__(self, args, pilot):
        self.pilot = pilot
        self.batch = Path(pilot['batch'])
        self.operator_id = ''
        self.record_type = 'practice'
        self.collect_meta = None
        self.substep_guard = None
        self.margin_guard = None
        super().__init__(args)
        self.label = VERSION + ':human_only'
        self.paused = True

    def bind(self):
        binding = load(self.source/'target-binding.json')
        self.identity_expected = dict(task=self.args.task,fixture_name=binding['fixture_name'],
            fixture_class=binding['fixture_class'],model_sha256=digest(self.source/'model.xml'))
        self.identity_context = dict(run_id=self.batch.name, group_id=self.pilot['scene_id'])
        self.native = observe_native(self, self.identity_expected)

    def freeze(self):
        if self.pilot.get('primary_enabled'):
            raise ValueError('Frozen primary Source cannot be overwritten')
        if any(c for c in self.trace()['contacts'] if any('gripper0_right_finger' in (c.get(n) or '') for n in ('geom1','geom2')) and any(self.identity_expected['fixture_name'] in (c.get(n) or '') for n in ('geom1','geom2'))):
            raise ValueError('Release and restore a precontact open target before saving Source')
        self.verify_target()
        result = super().freeze()
        self.bind()
        write_json(self.root/'current-source.json', dict(source=str(result), created_at=stamp(), config_version=self.pilot['config_version']))
        return result

    def begin(self):
        if hasattr(self, 'operator_entry'):
            self.operator_id = self.operator_entry.get().strip()
        validate_start(self.pilot, load_index(self.batch/'attempt-index.csv'), self.route, self.record_type, self.operator_id, self.source)
        self.restore()
        self.bind()
        guard = FormalSubstepMonitor(self,self.native['fixture_name'])
        guard.set_boundary(0, 'precontact')
        initial_contacts = list(guard.forbidden_contacts())
        if initial_contacts:
            raise ValueError('Source has forbidden native robot contact; see preparation receipt')
        super().begin()
        if hasattr(self,'operator_entry'):
            self.operator_entry.configure(state='disabled');self.panel.focus_set()
        # Freeze the evidence camera for the whole attempt; mouse view may change.
        self.substep_guard = FormalSubstepMonitor(self,self.native['fixture_name'])
        m,d = self.model_data()
        arm = self.robot.part_controllers['right']
        joints = [int(np.flatnonzero(m.jnt_qposadr==i)[0]) for i in arm.qpos_index]
        self.margin_guard = JointMarginMonitor(arm.qpos_index,m.jnt_range[joints],
            [mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) for j in joints])
        self.collect_meta = dict(version=VERSION, scene_id=self.pilot['scene_id'],
            scene_family_id=self.pilot['scene_family_id'], config_version=self.pilot['config_version'],
            source_id=self.source.name,record_type=self.record_type,operator_id=self.operator_id,
            route=self.route,code_commit=code_commit(),environment_seed=self.pilot['environment_seed'],
            max_sim_seconds=120,stop_budget='120 seconds of recorded simulation, all pilot routes',
            human_observation='',safety_status='pending',integrity_status='recording',route_semantics_status='pending',
            controller='unchanged keyboard key_actions',control_dt=.05,
            independent_cameras=True,reference_selected=False,formal_train_ready=False)
        self.recording['group'].attrs['collection_metadata'] = json.dumps(self.collect_meta)
        write_json(self.recording['path']/'collection-metadata.json',self.collect_meta)
        self.recording['events'].append(dict(step=0,event='collection_begin',operator_id=self.operator_id,record_type=self.record_type))
        self.base_drift_max = 0.
        self.locked_base_anchor = np.asarray(d.qpos[self.robot.part_controllers['base'].qpos_index]).copy()
        self.was_docked = False
        self.dock_observation = None

    def step(self, action, keys=()):
        if not self.recording:
            # Unrecorded practice stays explicit and never creates training labels.
            return super().step(action, keys)
        path = self.recording['path']; n = self.recording['n']
        m,d=self.model_data(); base=self.robot.part_controllers['base']
        if self.route=='D' and self.docked and not self.was_docked:
            self.locked_base_anchor=d.qpos[base.qpos_index].copy()
            self.dock_observation=dict(step=n,base_qpos=self.locked_base_anchor.tolist(),base_qvel=d.qvel[base.qvel_index].tolist(),selection='human-selected dock',requires_semantics_review=True)
            self.recording['events'].append(dict(event='docked_state',**self.dock_observation));self.was_docked=True
        if self.route=='E' or (self.route=='D' and self.docked):
            self.base_drift_max=max(self.base_drift_max,float(np.linalg.norm(d.qpos[base.qpos_index]-self.locked_base_anchor)))
        phase=phase_for(self.route,self.docked)
        self.substep_guard.set_boundary(n,phase)
        initial=self.integration().copy()
        try:
            with self.substep_guard:
                with GuardedIntegration(self.env.sim,d,self.margin_guard,lite_physics=self.env.lite_physics,step=n,phase=phase):
                    super().step(action,keys)
        except (FormalSafetyStop,JointMarginStop) as exc:
            np.savez_compressed(path/'partial-control-step.npz',initial_integration=initial,
                terminal_integration=self.integration(),attempted_action=action)
            write_json(path/'safety-stop.json',dict(at=stamp(),step=n,phase=phase,failure=exc.failure))
            self.finish('native_forbidden_contact_stop' if isinstance(exc,FormalSafetyStop) else 'joint_margin_stop')
        if self.recording and self.recording['n']>=2400:
            self.finish('human_budget_stop_120s')

    def finish(self, reason):
        if not self.recording:return
        path=self.recording['path']; n=self.recording['n']; start=self.recording['started']
        if self.route=='E' or (self.route=='D' and self.docked):
            _,d=self.model_data();base=self.robot.part_controllers['base']
            self.base_drift_max=max(self.base_drift_max,float(np.linalg.norm(d.qpos[base.qpos_index]-self.locked_base_anchor)))
        meta=dict(self.collect_meta or {})
        meta.update(stop_reason=reason,base_drift_max_generalized=getattr(self,'base_drift_max',0.),
                    human_selected_dock=getattr(self,'dock_observation',None),safety_status='not_passed' if (path/'safety-stop.json').exists() else 'pending',
                    note='Human outcome only; no automatic executor label; continuous clearance/human semantics still pending')
        if self.substep_guard is not None:
            write_json(path/'formal-native-substeps-receipt.json',self.substep_guard.save(path))
        if self.margin_guard is not None:
            # Mirror the existing guard receipt fields without changing limits.
            write_json(path/'joint-margin-monitor.json',self.margin_guard.receipt())
        if n:
            super().finish(reason)
            result=load(path/'result.json'); meta['integrity_status']='native_identity_verified'
        else:
            r=self.recording
            for key in ('h','trace','video','panoramic_video'):
                if r.get(key) is not None:r[key].close()
            result=dict(status='zero_full_steps',route=self.route,source=str(self.source),started_at=start,ended_at=stamp(),steps=0,
                checker_success=bool(self.env._check_success()),reason=reason,events=r['events'],review_status='pending',
                task_video_manifest=None,partial_control_step=(path/'partial-control-step.npz').exists())
            write_json(path/'result.json',result);write_json(path/'status.json',dict(status='zero_full_steps'))
            self.recording=None;self.last_attempt=path;self.paused=True;meta['integrity_status']='no_complete_control_frame'
        meta.update(ended_at=result['ended_at'],started_at=result['started_at'],steps=n,
            machine_success=result['checker_success'],route_semantics_status='pending',reference_selected=False)
        write_json(path/'collection-metadata.json',meta)
        append_index(self.batch/'attempt-index.csv',dict(attempt_id=path.name,source_id=self.source.name,config_version=self.pilot['config_version'],
            route=self.route,record_type=self.record_type,operator_id=self.operator_id,order=len(load_index(self.batch/'attempt-index.csv'))+1,
            started_at=result['started_at'],ended_at=result['ended_at'],machine_success=result['checker_success'],human_observation='',stop_reason=reason,
            safety_status=meta['safety_status'],integrity_status=meta['integrity_status'],route_semantics_status='pending',
            original_video_path=str(path/'original.mp4') if n else '',panoramic_video_path=str(path/'panoramic.mp4') if n else '',
            trajectory_path=str(path/'demo.hdf5'),result_path=str(path/'result.json'),reference_selected=False,
            notes=self.record_type+'; not an automatic strategy label'))
        self.message=f'SAVED {n} steps. {reason}. Safety/semantics review pending.'
        if hasattr(self,'operator_entry'):self.operator_entry.configure(state='normal')

    def create_panel(self):
        super().create_panel()
        import tkinter as tk
        self.panel.title(self.pilot['scene_id']+' | Human pilot')
        self.panel.geometry('410x1000+1200+0')
        tk.Label(self.panel,text='Operator ID (required before F2)',bg='#243447',fg='white').pack()
        self.operator_entry=tk.Entry(self.panel,font=('sans',14));self.operator_entry.pack(fill='x',padx=12)
        tk.Label(self.panel,text='PILOT / practice only\nF2 restores Source and records full attempt\nF4 = human contact mark, F3 = stop/save\nAuto-save after 10 successful steps',bg='#243447',fg='white',justify='left').pack(pady=8)

    def update_panel(self,mode=None):
        super().update_panel(mode)
        native=self.trace(); streak=self.recording['success_streak'] if self.recording else 0
        state='PAUSED' if self.paused else ('RECORDING' if self.recording else 'UNRECORDED PRACTICE')
        steps=self.recording['n'] if self.recording else 0
        self.status_label.config(text=f'{self.pilot["scene_id"]} | {state}\nHuman {self.route} | practice\nOpening {native["target"]["door"]:.5f}\nNative success: {native["success"]}\nHold: {streak}/10 | {steps*.05:.2f}/120s\n{self.message}')
        # The entry needs keyboard events; do not grab the keyboard until it loses focus.
        from Xlib import X
        entry_focus=self.panel.focus_get() is self.operator_entry
        if entry_focus and not getattr(self,'entry_ungrabbed',False):
            if not self.recording:self.paused=True
            self.keyboard_display.ungrab_keyboard(X.CurrentTime);self.keyboard_display.sync();self.entry_ungrabbed=True
        elif not entry_focus and getattr(self,'entry_ungrabbed',False):
            self.keyboard_display.screen().root.grab_keyboard(False,X.GrabModeAsync,X.GrabModeAsync,X.CurrentTime)
            self.keyboard_display.sync();self.keyboard.clear();self.entry_ungrabbed=False

    def replay(self):
        raise ValueError('Pilot UI does not launch dynamic replay; engineering review is separate')


def make_args(root,task,seed,source=None,interactive=False):
    return argparse.Namespace(output=str(root),task=task,layout=1,style=0,seed=seed,self_test=not interactive,
        source=str(source) if source else None,replay_attempt=None,resume_attempt=None,width=1280,height=720)


def prepare(batch,scene,task,seed):
    root=batch/'scenes'/scene/'pilot-v1';root.mkdir(parents=True,exist_ok=False)
    ref=Reference(make_args(root,task,seed));ref.label='human-scene-pilot-preparation'
    try:
        binding=load(root/'target-binding.json')
        initial=ref.trace()
        if initial['success']:raise ValueError('Initial target is already closed')
        guard=FormalSubstepMonitor(ref,binding['fixture_name']);bad=list(guard.forbidden_contacts())
        write_json(root/'initial-contact-check.json',dict(contacts=bad,phase='precontact',native_check_only=True,continuous_clearance='pending'))
        if bad:raise ValueError('Initial forbidden robot contact')
        source=ref.freeze()
        before=ref.integration().copy();rng=ref.env.rng.bit_generator.state
        ref.restore();error=float(np.max(abs(ref.integration()-before)))
        if error>1e-10 or rng!=ref.env.rng.bit_generator.state:raise ValueError('Saved Source restore mismatch')
        expected=dict(task=task,fixture_name=binding['fixture_name'],fixture_class=binding['fixture_class'],model_sha256=digest(source/'model.xml'),group_id=scene)
        native=observe_native(ref,expected)
        camera=prepare_camera(ref,expected,root/'camera-preview')
        candidates=[x for x in camera['candidates'] if x['minimum_visibility_ratio']>=1.]
        wide=max(candidates,key=lambda x:x['pixels']['base'])['camera'].copy();wide['distance']*=1.35
        ref.apply_camera(camera['camera']);Image.fromarray(ref.frame(camera['camera']).copy()).save(root/'main-preview.png')
        Image.fromarray(ref.frame(wide).copy()).save(root/'panoramic-preview.png');ref.apply_camera(camera['camera'])
        config=dict(version=VERSION,batch=str(batch),scene_id=scene,scene_family_id=scene+'-family',config_version='pilot-v1',
            task=task,environment_seed=seed,layout_id=1,style_id=0,source=str(source),created_at=stamp(),code_commit=code_commit(),
            main_camera=camera['camera'],panoramic_camera=wide,primary_enabled=False,frozen_at=None,route_order=['E','D','A'],
            intended_use='human practice pilots; outside main8/16 configuration count',max_sim_seconds=120,
            target_binding=str(source/'target-binding.json'),preview_visibility_pass=camera['visibility_pass'],
            restore_max_abs_error=error,source_rng_restored=True,new_task_actions=0,review_status='pending',formal_train_ready=False)
        write_json(root/'pilot.json',config);write_json(root/'native-identity.json',native)
        write_json(root/'restore-receipt.json',dict(created_at=stamp(),integration_max_abs_error=error,rng_identical=True,new_task_actions=0))
        print(json.dumps(dict(prepared=scene,source=str(source),config=str(root/'pilot.json'),opening=initial['target'],restore_error=error)),flush=True)
    finally:
        for name in ('observation_renderer','renderer'):
            obj=getattr(ref,name,None)
            if obj is not None:obj.close()
        ref.env.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','ui','record-smoke']);parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--scene',required=True);parser.add_argument('--task',choices=['CloseSingleDoor','CloseDrawer']);parser.add_argument('--seed',type=int)
    args=parser.parse_args();batch=args.batch.resolve()
    if args.mode=='prepare':
        return prepare(batch,args.scene,args.task,args.seed)
    config_path=batch/'scenes'/args.scene/'pilot-v1/pilot.json';config=load(config_path)
    source=Path(config['source']);output=batch/'episodes'/args.scene/'interactive'
    if args.mode=='record-smoke':output=batch/'episodes'/args.scene/'engineering-record-smoke'
    output.mkdir(parents=True,exist_ok=True)
    ref=PilotReference(make_args(output,config['task'],config['environment_seed'],source,interactive=args.mode=='ui'),config)
    try:
        ref.evidence_camera=config['main_camera'];ref.panoramic_camera=config['panoramic_camera'];ref.restore();ref.bind()
        ref.message='Pilot ready, PAUSED. Enter operator ID, click outside field, then F2.'
        write_json(output/'process.json',dict(started_at=stamp(),pid=os.getpid(),command=sys.argv,code_commit=code_commit(),python=sys.executable,
            CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),DISPLAY=os.environ.get('DISPLAY'),source=str(source),mode=args.mode))
        if args.mode=='record-smoke':
            # Small explicit engineering recording; never a human/main outcome.
            ref.pilot['allow_engineering_record_smoke']=True;ref.record_type='engineering_record_smoke'
            ref.operator_id='engineering_zero_input_smoke';ref.route='E';ref.begin()
            ref.recording['events'].append(dict(step=0,event='engineering_smoke_not_human'))
            action=ref.robot.create_action_vector(key_actions(set(),False,True))
            for _ in range(2):
                if ref.recording:ref.step(action)
            if ref.recording:ref.finish('engineering_record_smoke')
            print('RECORD_SMOKE_DONE',ref.last_attempt,flush=True)
        else:
            ref.run()
    finally:
        if ref.recording:ref.finish('process_exit')
        for name in ('observation_renderer','renderer'):
            obj=getattr(ref,name,None)
            if obj is not None:obj.close()
        ref.env.close()


if __name__=='__main__':main()
