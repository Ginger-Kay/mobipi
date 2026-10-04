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
from teleop_reference import Reference, write_json, stamp, key_actions, Keyboard
from mobiwam.human_scene_collection import VERSION, validate_start, append_index, load_index, phase_for
from mobiwam.task_video_identity import observe_native
from mobiwam.dr_v04_r2 import prepare_camera
from mobiwam.reference_formal_substep import FormalSubstepMonitor, FormalSafetyStop
from mobiwam.reference_prefix_safety import JointMarginMonitor, JointMarginStop, GuardedIntegration
from mobiwam.teleop_feedback import signed_planar_angle_deg, touching_fingers


def load(p): return json.loads(Path(p).read_text())
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def code_commit():
    return subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[1]), 'rev-parse', 'HEAD'], text=True).strip()


def restore_saved_integration(ref):
    Reference.restore(ref)
    # mj_forward recomputes the solver warm-start cache. Restore the saved cache
    # after derived kinematics/controller refresh, without advancing physics.
    m,d=ref.model_data();saved=np.load(ref.source/'integration.npy')
    scratch=mujoco.MjData(m);mujoco.mj_setState(m,scratch,saved,ref.kind)
    warmstart_delta=float(np.max(abs(d.qacc_warmstart-scratch.qacc_warmstart),initial=0))
    d.qacc_warmstart[:]=scratch.qacc_warmstart
    error=float(np.max(abs(ref.integration()-saved),initial=0))
    if error>1e-10:raise ValueError(f'Source integration restore mismatch after warm-start restore: {error}')
    ref.restore_receipt=dict(max_abs_error=error,solver_warmstart_recomputed_delta=warmstart_delta,
        solver_warmstart_restored=True,new_physics_steps=0)
    return ref.restore_receipt


class PilotReference(Reference):
    def __init__(self, args, pilot):
        self.pilot = pilot
        self.batch = Path(pilot['batch'])
        self.operator_id = ''
        self.record_type = 'practice'
        self.collect_meta = None
        self.substep_guard = None
        self.margin_guard = None
        interactive=not args.self_test
        # This VNC server has no GLX extension. Use the existing EGL renderer
        # for a Tk display; task inputs, policy cameras and physics stay intact.
        args.self_test=True
        super().__init__(args)
        if interactive:self.keyboard=Keyboard()
        self.label = VERSION + ':human_only'
        self.paused = True

    def restore(self):
        self._live_frame_cache=None
        self._displayed_frame_key=None
        return restore_saved_integration(self)

    def frame(self,camera=None):
        # Reuse the already recorded main-camera RGB only for screen display.
        # The base recorder still renders/validates every original video frame.
        rgb=super().frame(camera)
        m,d=self.model_data()
        actual=camera if camera is not None else self.camera_state()
        self._live_frame_cache=((id(m),float(d.time),json.dumps(actual,sort_keys=True)),rgb.copy())
        return rgb

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

    def prepare_live_action(self, parts):
        if self.pilot.get('paired_protocol_version')!='human-eda-v2-stowed':
            return parts
        from mobiwam.human_paired_protocol import route_inputs
        if self.route=='D' and not self.docked and self.keyboard:
            self.keyboard.grasp=False
        return route_inputs(parts,self.route,self.docked)

    def mark_event(self, cmd):
        if self.pilot.get('paired_protocol_version')=='human-eda-v2-stowed':
            if cmd=='f4' and self.route=='D' and not self.docked:
                raise ValueError('D: dock with F5 before approach/contact')
            if cmd=='f5':
                from mobiwam.human_paired_protocol import validate_dock
                m,d=self.model_data();base=self.robot.part_controllers['base'];arm=self.robot.part_controllers['right']
                target=self.native['fixture_name']
                contact=any(any((c.get(k) or '').startswith(target) for k in ('geom1','geom2'))
                    and any((c.get(k) or '').startswith(('robot0_','gripper0_','mobilebase0_')) for k in ('geom1','geom2'))
                    for c in self.trace()['contacts'])
                validate_dock(self.route,self.docked,d.qvel[base.qvel_index],d.qpos[arm.qpos_index],
                              self.pilot['common_stow_qpos'],bool(self.keyboard and self.keyboard.grasp),contact)
        return super().mark_event(cmd)

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
        self.primary_reservation = None
        if self.record_type == 'primary':
            from mobiwam.human_primary import reserve_primary
            self.primary_reservation = reserve_primary(self.pilot, self.route, self.operator_id)
        super().begin(restore_source=False)
        if self.primary_reservation:
            reservation=load(self.primary_reservation)
            reservation.update(status='recording',attempt=str(self.recording['path']))
            write_json(self.primary_reservation,reservation)
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
        self.collect_meta['source_lineage'] = self.pilot.get('source_lineage')
        self.collect_meta['freeze_receipt'] = self.pilot.get('freeze_receipt')
        self.collect_meta['primary_reservation'] = str(self.primary_reservation) if self.primary_reservation else None
        if self.pilot.get('paired_protocol_version')=='human-eda-v2-stowed':
            self.recording['initial_stow_required']=self.route=='D'
            self.collect_meta['paired_protocol_version']='human-eda-v2-stowed'
            self.collect_meta['common_stow_qpos']=self.pilot['common_stow_qpos']
        self.recording_wall_started = time.monotonic()
        self.recording['group'].attrs['collection_metadata'] = json.dumps(self.collect_meta)
        write_json(self.recording['path']/'collection-metadata.json',self.collect_meta)
        self.recording['events'].append(dict(step=0,event='collection_begin',operator_id=self.operator_id,record_type=self.record_type))
        self.base_drift_max = 0.
        self.locked_base_anchor = np.asarray(d.qpos[self.robot.part_controllers['base'].qpos_index]).copy()
        self.was_docked = False
        self.dock_observation = None

    def step(self, action, keys=()):
        if not self.recording:
            if self.record_type == 'primary':
                self.paused=True
                self.message='PRIMARY: press F2 to restore and start the recorded attempt.'
                return
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
        if getattr(self,'primary_reservation',None):
            reservation=load(self.primary_reservation)
            reservation.update(status='completed',attempt=str(path),ended_at=result['ended_at'],stop_reason=reason)
            write_json(self.primary_reservation,reservation)
        self.message=f'SAVED {n} steps. {reason}. Safety/semantics review pending.'
        if reason == 'joint_margin_stop':
            failure=load(path/'safety-stop.json')['failure']
            self.message=(f'SAFETY STOP: {failure["joint"]} near limit '
                          f'({failure["margin_rad"]:.4f} rad). NOT a timeout. '
                          'Saved. Next attempt: reposition base or reverse arm motion.')
        elif reason == 'native_forbidden_contact_stop':
            self.message='SAFETY STOP: forbidden contact. Saved; inspect contact before another attempt.'
        if hasattr(self,'operator_entry'):self.operator_entry.configure(state='normal')

    def create_panel(self):
        super().create_panel()
        import tkinter as tk
        self.panel.title(self.pilot['scene_id']+' | Human '+self.record_type)
        if self.pilot.get('paired_protocol_version')=='human-eda-v2-stowed':
            tk.Label(self.panel,text=f'COMMON STOWED SOURCE / {self.record_type.upper()}\nD: base only until F5; arm and closing blocked.\nRelease base keys, wait stopped, then F5.\nAfter F5: base locked; approach with arm.\nE: base always locked. A: both available.',
                     bg='#243447',fg='#ffe08a',justify='left').pack()

        self.panel.geometry('410x1000+1200+0')
        for widget in self.panel.winfo_children():
            if isinstance(widget,tk.Label) and str(widget.cget('text')).startswith('ARM:'):
                widget.configure(text='MOVE ARM: arrows = horizontal shift\n; / . = raise / lower\nTURN GRIPPER: O / P = yaw + / -\nO counterclockwise, P clockwise (top view)\nY / H = pitch, E / R = roll\nMOVE BASE: W/A/S/D; Z/X turns BASE\nSpace = open / close gripper\n\nMove along the door arc AND turn the wrist.\nTap keys briefly; Esc pauses to inspect.'.replace('Move along the door arc AND turn the wrist.', 'Drawer: keep wrist aligned; slide straight in.' if self.native['fixture_class']=='Drawer' else 'Move along the door arc AND turn the wrist.'),
                    fg='#ffe08a',font=('sans',12))
        tk.Label(self.panel,text='Operator ID (required before F2)',bg='#243447',fg='white').pack()
        self.operator_entry=tk.Entry(self.panel,font=('sans',14));self.operator_entry.pack(fill='x',padx=12)
        self.operator_entry.bind('<Button-1>',lambda event:self.operator_entry.focus_force())
        tk.Label(self.panel,text=f'{self.record_type.upper()} | order: {" -> ".join(self.pilot["route_order"])}\nF2 restores Source and records full attempt\nF4 = human contact mark, F3 = stop/save\nEsc: pause to think (no simulated time passes)\nAuto-save: success, safety stop, or 120 SIM seconds',bg='#243447',fg='white',justify='left').pack(pady=8)

        self.live_window=tk.Toplevel(self.panel);self.live_window.title(self.pilot['scene_id']+' | Live native view')
        self.live_window.geometry('1200x820+0+0');self.live_window.protocol('WM_DELETE_WINDOW',lambda:self.keyboard.commands.put('pause'))
        self.live_label=tk.Label(self.live_window,bg='black');self.live_label.pack()
        self.live_camera=json.loads(json.dumps(self.evidence_camera));self.last_live_frame=0.;self.last_ui_status=0.
        tk.Label(self.live_window,text='Mouse drag: view angle | Wheel: zoom | Buttons below: view only. Recording uses fixed target + panorama cameras.').pack()
        self.contact_label=tk.Label(self.live_window,text='',font=('sans',14),justify='left',wraplength=1150)
        self.contact_label.pack(fill='x')
        def select_camera(camera):self.live_camera=json.loads(json.dumps(camera));self.last_live_frame=0.
        tk.Button(self.live_window,text='Target view',command=lambda:select_camera(self.evidence_camera)).pack(side='left')
        tk.Button(self.live_window,text='Base overview',command=lambda:select_camera(self.panoramic_camera)).pack(side='left')
        if self.native['fixture_class'] in ('Microwave','Drawer'):
            def handle_closeup():
                m,d=self.model_data()
                hid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,self.native['fixture_name']+('_door_handle_handle' if self.native['fixture_class']=='Drawer' else '_door_handle'))
                if hid>=0:
                    camera=dict(self.evidence_camera,lookat=d.geom_xpos[hid].tolist(),distance=.85)
                    select_camera(camera)
            tk.Button(self.live_window,text='Handle close-up (recenter)',command=handle_closeup).pack(side='left')
        self.camera_drag=None
        def drag_start(event):self.camera_drag=(event.x,event.y)
        def drag(event):
            if self.camera_drag is None:return
            x,y=self.camera_drag;self.camera_drag=(event.x,event.y)
            self.live_camera['azimuth']+=(event.x-x)*.3
            self.live_camera['elevation']=float(np.clip(self.live_camera['elevation']+(event.y-y)*.3,-80,10))
            self.last_live_frame=0.
        def zoom(factor):self.live_camera['distance']=float(np.clip(self.live_camera['distance']*factor,.5,8));self.last_live_frame=0.
        self.live_label.bind('<Button-1>',drag_start);self.live_label.bind('<B1-Motion>',drag)
        self.live_label.bind('<Button-4>',lambda event:zoom(.9));self.live_label.bind('<Button-5>',lambda event:zoom(1.1))
        self.panel.focus_force()

    def update_panel(self,mode=None):
        while self.keyboard_display.pending_events():self.keyboard_display.next_event()
        native=self.trace(); streak=self.recording['success_streak'] if self.recording else 0
        state='PAUSED' if self.paused else ('RECORDING' if self.recording else 'UNRECORDED PRACTICE')
        steps=self.recording['n'] if self.recording else 0
        m,d=self.model_data()
        arm=self.robot.part_controllers['right']
        joints=[int(np.flatnonzero(m.jnt_qposadr==i)[0]) for i in arm.qpos_index]
        limits=m.jnt_range[joints];q=d.qpos[arm.qpos_index]
        margins=np.minimum(q-limits[:,0],limits[:,1]-q);j=int(np.argmin(margins))
        margin=float(margins[j]);joint=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,joints[j])
        wall=time.monotonic()-getattr(self,'recording_wall_started',time.monotonic()) if self.recording else 0.
        caution='\nNEAR LIMIT: reverse arm / reposition base' if margin<.15 else ''
        yaw_gap=None;turn_text='';finger_contacts=None
        grip_closed=bool(self.keyboard and self.keyboard.grasp)
        if self.native['fixture_class']=='Microwave':
            hid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,self.native['fixture_name']+'_door_handle')
            sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'gripper0_right_grip_site')
            if min(hid,sid)>=0:
                yaw_gap=signed_planar_angle_deg(d.site_xmat[sid].reshape(3,3)[:,2],d.geom_xmat[hid].reshape(3,3)[:,1])
            turn_text=('\nDoor/wrist yaw gap: tilted' if yaw_gap is None else
                       f'\nDoor/wrist yaw gap: {yaw_gap:+.1f} deg\nO turns + | P turns - (top view)')
            geoms=self.robot.gripper['right'].important_geoms
            finger_contacts=touching_fingers(native['contacts'],self.native['fixture_name']+'_door_handle',
                                             (geoms['left_finger'],geoms['right_finger']))
            count=sum(finger_contacts)
            warning=grip_closed and count<2
            contact_text=(f'Handle contact: {count}/2 fingers | command: {"CLOSE" if grip_closed else "OPEN"}'
                          '\nSampled contact only; not proof of secure grasp. '+
                          ('Fewer than 2: Esc to inspect before moving.' if warning else 'Watch both fingers as the door turns.'))
            self.contact_label.config(text=contact_text,fg='#9c3100' if warning else '#16364a')
        if self.native['fixture_class']=='Drawer':
            geoms=self.robot.gripper['right'].important_geoms
            finger_contacts=touching_fingers(native['contacts'],self.native['fixture_name']+'_door_handle_handle',
                                             (geoms['left_finger'],geoms['right_finger']))
            count=sum(finger_contacts)
            warning=grip_closed and count<2
            self.contact_label.config(
                text=f'Handle contact: {count}/2 fingers | command: {"CLOSE" if grip_closed else "OPEN"}\n' +
                     ('Common stowed start: align fingers above/below handle after docking.\n' if self.pilot.get('paired_protocol_version')=='human-eda-v2-stowed' else 'Start is aligned: keep fingers above/below the horizontal handle. Approach with small translations.\n') +
                     'Sampled contact only; not proof of secure grasp. Esc pauses; Space closes/opens.',
                fg='#9c3100' if warning else '#16364a')
        self.status_label.config(fg='#ffbf47' if margin<.15 else 'white',
            text=f'{self.pilot["scene_id"]} | {state}\nHuman {self.route} | {self.record_type}\nOpening {native["target"]["door"]:.5f}\nNative success: {native["success"]}\nHold: {streak}/10 | SIM {steps*.05:.2f}/120s\nWALL {wall:.0f}s | joint margin {margin:.3f} rad\n{joint}{caution}{turn_text}\n{self.message}')
        now=time.monotonic()
        m,d=self.model_data()
        view_key=(id(m),float(d.time),json.dumps(self.live_camera,sort_keys=True))
        if now-self.last_live_frame>=.05 and view_key!=getattr(self,'_displayed_frame_key',None):
            from PIL import ImageTk
            cache=getattr(self,'_live_frame_cache',None)
            rgb=cache[1] if cache is not None and cache[0]==view_key else self.frame(self.live_camera)
            self.live_photo=ImageTk.PhotoImage(Image.fromarray(rgb).resize((1200,675)))
            self.live_label.configure(image=self.live_photo);self.last_live_frame=now
            self._displayed_frame_key=view_key
        if now-self.last_ui_status>=5:
            write_json(self.root/'ui-status.json',dict(at=stamp(),pid=os.getpid(),paused=self.paused,recording=bool(self.recording),
                scene_id=self.pilot['scene_id'],source=str(self.source),route=self.route,record_type=self.record_type,
                config_version=self.pilot['config_version'],frozen_at=self.pilot.get('frozen_at'),sim_time=native['sim_time'],
                opening=native['target'],native_success=native['success'],operator_id=self.operator_id,
                complete_control_steps=steps,keyboard_entry_focused=self.panel.focus_get() is self.operator_entry,
                minimum_arm_joint_margin_rad=margin,nearest_limit_joint=joint,wall_seconds=wall,
                door_wrist_planar_yaw_gap_deg=yaw_gap,handle_finger_contacts=finger_contacts,
                gripper_close_command=grip_closed,live_camera=self.live_camera))
            self.last_ui_status=now
        self.panel.update()
        self.panel.lift()
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


def prepare(batch,scene,task,seed,resume_source=None):
    root=batch/'scenes'/scene/'pilot-v1'
    if resume_source:
        if not root.exists() or (root/'pilot.json').exists():raise ValueError('Resume requires incomplete preparation')
        resume_source=Path(resume_source).resolve()
        if resume_source.parent!=root.resolve():raise ValueError('Resume Source outside this pilot')
        output=root/('recovery-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()))
    else:
        root.mkdir(parents=True,exist_ok=False);output=root
    ref=Reference(make_args(output,task,seed,resume_source));ref.label='human-scene-pilot-preparation'
    if resume_source:restore_saved_integration(ref)
    try:
        binding=load(ref.root/'target-binding.json')
        initial=ref.trace()
        if initial['success']:raise ValueError('Initial target is already closed')
        guard=FormalSubstepMonitor(ref,binding['fixture_name']);bad=list(guard.forbidden_contacts())
        write_json(root/'initial-contact-check.json',dict(contacts=bad,phase='precontact',native_check_only=True,continuous_clearance='pending'))
        if bad:raise ValueError('Initial forbidden robot contact')
        source=ref.source if resume_source else ref.freeze()
        before=ref.integration().copy();rng=ref.env.rng.bit_generator.state
        restore_saved_integration(ref);error=float(np.max(abs(ref.integration()-before)))
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
        write_json(root/'restore-receipt.json',dict(created_at=stamp(),integration_max_abs_error=error,rng_identical=True,new_task_actions=0,solver_state=ref.restore_receipt))
        print(json.dumps(dict(prepared=scene,source=str(source),config=str(root/'pilot.json'),opening=initial['target'],restore_error=error)),flush=True)
    finally:
        for name in ('observation_renderer','renderer'):
            obj=getattr(ref,name,None)
            if obj is not None:obj.close()
        ref.env.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','ui','record-smoke']);parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--scene',required=True);parser.add_argument('--task',choices=['CloseSingleDoor','CloseDrawer']);parser.add_argument('--seed',type=int);parser.add_argument('--resume-source',type=Path)
    parser.add_argument('--route',choices=['A','E','D'],default='A',help='Initial human route; switching remains explicit')
    parser.add_argument('--record-type',choices=['practice','primary'],default='practice')
    parser.add_argument('--config',type=Path,help='Explicit versioned pilot config; original Source/config remains immutable')
    args=parser.parse_args();batch=args.batch.resolve()
    if args.mode=='prepare':
        return prepare(batch,args.scene,args.task,args.seed,args.resume_source)
    config_path=args.config or batch/'scenes'/args.scene/'pilot-v1/pilot.json';config=load(config_path)
    if config['scene_id'] != args.scene or Path(config['batch']).resolve() != batch:
        raise ValueError('Pilot config scene/batch differs from launch')
    if args.record_type=='primary':
        if args.mode!='ui':raise ValueError('Primary is human UI only')
        from mobiwam.human_primary import verify_freeze
        verify_freeze(config)
    source=Path(config['source']);output=batch/'episodes'/args.scene/'interactive'
    if args.mode=='record-smoke':output=batch/'episodes'/args.scene/'engineering-record-smoke'
    output.mkdir(parents=True,exist_ok=True)
    ref=PilotReference(make_args(output,config['task'],config['environment_seed'],source,interactive=args.mode=='ui'),config)
    try:
        ref.evidence_camera=config['main_camera'];ref.panoramic_camera=config['panoramic_camera'];ref.restore();ref.bind();ref.route=args.route;ref.record_type=args.record_type
        ref.message=f'{ref.record_type.upper()} ready, PAUSED. Enter operator ID, click outside field, then F2.'
        write_json(output/'process.json',dict(started_at=stamp(),pid=os.getpid(),command=sys.argv,code_commit=code_commit(),python=sys.executable,
            CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),DISPLAY=os.environ.get('DISPLAY'),source=str(source),mode=args.mode,
            config_path=str(config_path.resolve()),config_version=config['config_version']))
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
