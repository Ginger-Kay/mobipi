"""Keyboard human-reference collection for the pinned RoboCasa fork.

Practice first. F1 freezes a development source, F2 records from that source,
F3 finishes, F4 marks contact onset, F5 marks docking (locks D base),
F6 selects A/E/D in practice, F8 restores source, F9 replays last attempt.
All attempts remain references requiring human collision/cooperation review.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import queue
import threading
import time

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
import robocasa
import robosuite
from robosuite.controllers import load_composite_controller_config


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def identifier():
    return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.tolist()) + '\n')


def key_actions(keys, grasp, base_locked=False):
    """Normalized OSC delta and current-base-frame base velocity inputs.

    Arm signs reproduce Keyboard + mirror_actions=True in the current fork.
    Base controller swaps its first two raw inputs internally (recorded schema).
    """
    axis = lambda plus, minus: float(plus in keys) - float(minus in keys)
    arm = np.array([axis('up', 'down'), axis('left', 'right'),
                    axis(';', '.'), axis('e', 'r'), axis('y', 'h'),
                    axis('o', 'p')]) * np.array([.06, .06, .06, .06, .06, .06])
    # The fork has 250 N slide friction and kv=1000: tiny commands below
    # ~0.25 stall rather than producing slow motion. Keep physics unchanged.
    base = np.array([axis('w', 's'), axis('d', 'a'), axis('z', 'x')]) * np.array([.35, .35, .20])
    if base_locked:
        base[:] = 0
    return {'right': arm, 'right_gripper': np.array([1. if grasp else -1.]),
            'base': base, 'base_mode': 1. if np.any(base) else -1.}


class Keyboard:
    def __init__(self):
        from pynput.keyboard import Listener
        self.keys = set()
        self.grasp = False
        self.commands = queue.Queue()
        self.lock = threading.Lock()
        self.listener = Listener(on_press=self.press, on_release=self.release)
        self.listener.start()

    @staticmethod
    def name(key):
        return getattr(key, 'char', None) or getattr(key, 'name', '')

    def press(self, key):
        name = self.name(key)
        with self.lock:
            fresh = name not in self.keys
            self.keys.add(name)
            if fresh and name.startswith('f') and name[1:].isdigit():
                self.commands.put(name)
            if fresh and name == 'esc':
                self.commands.put('pause')
            if fresh and name == 'space':
                self.grasp = not self.grasp

    def release(self, key):
        with self.lock:
            self.keys.discard(self.name(key))

    def clear(self):
        with self.lock:
            self.keys.clear()

    def read(self, locked=False):
        with self.lock:
            keys = set(self.keys)
            return key_actions(keys, self.grasp, locked), sorted(keys)


class ReplayPreview:
    """Display rendered replay frames without exposing a physics viewer."""

    def __init__(self, panel):
        import tkinter as tk
        self.window = tk.Toplevel(panel)
        self.window.title('MobiWAM replay (view only)')
        self.window.geometry('1200x900+0+0')
        self.window.protocol('WM_DELETE_WINDOW', lambda: None)
        self.label = tk.Label(self.window, bg='black')
        self.label.pack(fill='both', expand=True)

    def show(self, frame):
        from PIL import Image, ImageTk
        img = Image.fromarray(frame)
        img.thumbnail((1200, 900))
        self.image = ImageTk.PhotoImage(img, master=self.window)
        self.label.configure(image=self.image)

    def close(self):
        self.window.destroy()


class Reference:
    def __init__(self, args):
        self.args = args
        self.label = 'engineering_self_test' if args.self_test else 'human_teleoperation_reference'
        self.root = Path(args.output)
        self.root.mkdir(parents=True, exist_ok=True)
        controller = load_composite_controller_config(robot='PandaOmron')
        self.config = dict(env_name=args.task, robots='PandaOmron',
                           controller_configs=controller, layout_ids=[args.layout],
                           style_ids=[args.style], seed=args.seed,
                           translucent_robot=False, control_freq=20,
                           use_camera_obs=False, has_renderer=not args.self_test,
                           has_offscreen_renderer=False, renderer='mjviewer',
                           render_camera=None, ignore_done=True,
                           renderer_config={'cam_config': {'lookat': [0, 0, 1],
                               'distance': 4.3, 'azimuth': 180, 'elevation': -20}})
        if args.task == 'CloseSingleDoor':
            from robocasa.models.fixtures import FixtureType
            self.config['door_id'] = int(FixtureType.MICROWAVE)
        requested_source = Path(args.source) if args.source else None
        if args.replay_attempt:
            requested_source = Path(args.replay_attempt).resolve().parent.parent
        elif args.resume_attempt:
            requested_source = Path(args.resume_attempt).resolve().parent.parent
        if requested_source:
            self.config = json.loads((requested_source.parent / 'env_config.json').read_text())
            self.config['has_renderer'] = not (args.self_test or args.replay_attempt)
            args.task = self.config['env_name']
        write_json(self.root / 'env_config.json', self.config)
        self.env = robosuite.make(**self.config)
        self.env.reset()
        self.robot = self.env.robots[0]
        assert self.robot.composite_controller.name == 'HYBRID_MOBILE_BASE'
        assert self.robot.part_controllers['right'].input_type == 'delta'
        self.source = requested_source
        self.recording = None
        last_attempt = args.replay_attempt or args.resume_attempt
        self.last_attempt = Path(last_attempt) if last_attempt else None
        self.route = 'A'
        self.docked = False
        self.renderer = None
        self.observation_renderer = None
        self.policy_cameras = ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand']
        self.render_options = mujoco.MjvOption()
        self.render_options.geomgroup[0] = 0  # match viewer: hide collision meshes
        self.capture_camera = mujoco.MjvCamera()
        self.capture_camera.lookat[:] = [0, 0, 1]
        self.capture_camera.distance = 4.3
        self.capture_camera.azimuth = 180
        self.capture_camera.elevation = -20
        # Kitchen replaces renderer_config with its actual layout camera.
        # Use that world-frame camera for native video too, not the generic
        # [0,0,1] viewer fallback which can sit behind the cabinets.
        camera = dict(self.env.renderer_config['cam_config'])
        camera['lookat'] = list(camera['lookat'])
        camera['lookat'][2] = 1.0
        camera['distance'] *= 1.3
        self.env.renderer_config = {'cam_config': camera}
        self.apply_camera(camera)
        self.keyboard = None if (args.self_test or args.replay_attempt) else Keyboard()
        self.paused = bool(requested_source)
        self.panel = None
        self.message = 'Practice only. No data is being recorded.'
        self.base_body = self.robot.robot_model.base.root_body
        self.kind = mujoco.mjtState.mjSTATE_INTEGRATION
        schema = {}
        for name, controller in self.robot.part_controllers.items():
            schema[name] = {k: getattr(controller, k) for k in
                           ('input_type', 'input_ref_frame', 'input_min', 'input_max',
                            'output_min', 'output_max') if hasattr(controller, k)}
        write_json(self.root / 'action_schema.json', {
            'splits': self.robot.composite_controller._action_split_indexes,
            'last_dimension': 'base_mode: +1 moving-base desired-goal update; -1 achieved-goal update',
            'part_controllers': schema, 'base_controller_note':
            'Raw inputs 0/1 are swapped internally and transformed using current vs initial base yaw; raw commands are not world-frame displacement.',
            'control_dt': .05, 'label': self.label,
            'formal_obc_split': 'excluded', 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'versions': {'robocasa': robocasa.__version__, 'robosuite': robosuite.__version__, 'mujoco': mujoco.__version__}})
        self.verify_target()

    def verify_target(self):
        fixture = self.env.drawer if self.args.task == 'CloseDrawer' else self.env.door_fxtr
        if self.args.task == 'CloseSingleDoor':
            from robocasa.models.fixtures import Microwave
            assert isinstance(fixture, Microwave), 'CloseSingleDoor must target the microwave for this collection'
        m, d = self.model_data()
        joints = []
        for i in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or ''
            if name.startswith(fixture.name + '_'):
                joints.append({'name': name, 'joint_id': i, 'qpos_address': int(m.jnt_qposadr[i]),
                               'qpos': float(d.qpos[m.jnt_qposadr[i]]), 'range': m.jnt_range[i].copy()})
        assert joints, 'No target joint found'
        if self.args.task == 'CloseSingleDoor':
            # This pinned fork indexes qpos by joint id in Microwave.get_door_state.
            # Fail closed if the model ordering would make that read another joint.
            hinge = next(j for j in joints if j['name'].endswith('_microjoint'))
            assert hinge['joint_id'] == hinge['qpos_address'], 'Microwave checker joint indexing mismatch'
        write_json(self.root / 'target-binding.json', {'verified_at': stamp(),
            'task': self.args.task, 'fixture_name': fixture.name, 'fixture_class': type(fixture).__name__,
            'language': self.env.get_ep_meta()['lang'], 'joints': joints,
            'opening': fixture.get_door_state(env=self.env), 'checker_success': bool(self.env._check_success()),
            'formal_train_ready': False, 'scope': 'development human references'})
        self.target_name = 'DRAWER' if self.args.task == 'CloseDrawer' else 'MICROWAVE DOOR'

    def append_observation(self, group):
        """State-aligned raw observations, independent of the free viewing camera."""
        values = dict(self.env._get_observations(force_update=True))
        m, d = self.model_data()
        if self.observation_renderer is None:
            self.observation_renderer = mujoco.Renderer(m, height=256, width=256)
        poses = []
        for camera in self.policy_cameras:
            cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            assert cid >= 0, f'Missing policy camera {camera}'
            self.observation_renderer.update_scene(d, camera=camera, scene_option=self.render_options)
            values[camera + '_image'] = self.observation_renderer.render().copy()
            pose = np.eye(4)
            pose[:3, :3] = d.cam_xmat[cid].reshape(3, 3)
            pose[:3, 3] = d.cam_xpos[cid]
            poses.append(pose)
        values['camera_to_world'] = np.asarray(poses)
        values['sim_time'] = np.asarray(d.time)
        obs = group.require_group('obs')
        for key, value in values.items():
            value = np.asarray(value)
            if value.dtype.kind not in 'buif':
                continue
            if key not in obs:
                obs.create_dataset(key, shape=(0,) + value.shape, maxshape=(None,) + value.shape,
                    chunks=(1,) + value.shape, dtype=value.dtype, compression='lzf')
            ds = obs[key]
            ds.resize(ds.shape[0] + 1, axis=0)
            ds[-1] = value
        obs.attrs['alignment'] = 'obs[t] and states[t] precede actions[t]; includes final obs[T]'
        obs.attrs['image_convention'] = 'RGB uint8 HWC top-left origin; native mujoco.Renderer'
        obs.attrs['camera_names'] = json.dumps(self.policy_cameras)
        obs.attrs['camera_fovy_degrees'] = [float(m.cam_fovy[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, c)]) for c in self.policy_cameras]
        obs.attrs['history'] = 'contiguous 20Hz observations; no free-view camera inputs'

    def model_data(self):
        return self.env.sim.model._model, self.env.sim.data._data

    def apply_camera(self, camera):
        self.capture_camera.lookat[:] = camera['lookat']
        for name in ('distance', 'azimuth', 'elevation'):
            setattr(self.capture_camera, name, camera[name])
        self.env.renderer_config = {'cam_config': camera}
        if self.env.viewer:
            self.env.viewer.camera_config = camera

    def camera_state(self):
        camera = self.capture_camera
        if self.env.viewer and self.env.viewer.viewer:
            camera = self.env.viewer.viewer.cam
        return {'lookat': list(camera.lookat), 'distance': camera.distance,
                'azimuth': camera.azimuth, 'elevation': camera.elevation}

    def integration(self):
        m, d = self.model_data()
        state = np.empty(mujoco.mj_stateSize(m, self.kind))
        mujoco.mj_getState(m, d, state, self.kind)
        return state

    def trace(self):
        m, d = self.model_data()
        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.base_body)
        fixture = self.env.drawer if self.args.task == 'CloseDrawer' else self.env.door_fxtr
        joints = fixture.get_door_state(env=self.env)
        contacts = [{'geom1': mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                     'geom2': mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2),
                     'distance': float(c.dist)} for c in d.contact[:d.ncon]]
        arm = self.robot.part_controllers['right']
        return {'sim_time': float(d.time), 'base_pos': d.xpos[base].copy(),
                'base_quat': d.xquat[base].copy(), 'qpos': d.qpos.copy(),
                'qvel': d.qvel.copy(), 'arm_qpos': d.qpos[arm.qpos_index].copy(),
                'eef_pos': arm.ref_pos.copy(),
                'target': joints, 'contacts': contacts,
                'success': bool(self.env._check_success())}

    def freeze(self):
        if self.env._check_success():
            raise ValueError('Source is already successful; use a pre-contact open target.')
        source = self.root / ('source-' + identifier())
        source.mkdir()
        (source / 'model.xml').write_text(self.env.sim.model.get_xml())
        (source / 'target-binding.json').write_text((self.root / 'target-binding.json').read_text())
        np.save(source / 'integration.npy', self.integration())
        write_json(source / 'ep_meta.json', self.env.get_ep_meta())
        write_json(source / 'rng.json', self.env.rng.bit_generator.state)
        write_json(source / 'source.json', {'created_at': stamp(), 'trace': self.trace(),
            'grasp_command': self.keyboard.grasp if self.keyboard else False,
            'camera': self.camera_state(),
            'label': self.label, 'precontact_and_collision_review': 'pending',
            'restore': 'reset XML; restore full mjSTATE_INTEGRATION; forward; refresh controller origins/states; reset goals then initialize OSC zero-delta achieved goals in its configured frame; no unlogged physics steps'})
        self.source = source
        self.message = 'Source saved. Record always restores this source.'
        print('SOURCE SAVED', source, flush=True)
        return source

    def restore(self):
        if self.observation_renderer is not None:
            self.observation_renderer.close()
            self.observation_renderer = None
        assert self.source is not None, 'Press F1 to save a source first.'
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.env.set_ep_meta(json.loads((self.source / 'ep_meta.json').read_text()))
        self.env.reset_from_xml_string((self.source / 'model.xml').read_text())
        self.robot = self.env.robots[0]
        m, d = self.model_data()
        mujoco.mj_setState(m, d, np.load(self.source / 'integration.npy'), self.kind)
        mujoco.mj_forward(m, d)
        self.env.rng.bit_generator.state = json.loads((self.source / 'rng.json').read_text())
        self.robot.composite_controller.update_state()
        for c in self.robot.part_controllers.values():
            try:
                c.update(force=True)
            except TypeError:
                c.update()
        self.robot.composite_controller.reset()
        # This fork's OSC reset_goal stores world-frame ref_pos even when its
        # configured input frame is base. Initialize through set_goal instead
        # before a positive base_mode selects desired-goal accumulation.
        arm = self.robot.part_controllers['right']
        arm.set_goal_update_mode('achieved')
        arm.set_goal(np.zeros(arm.control_dim))
        self.docked = False
        if self.keyboard:
            self.keyboard.clear()
            self.keyboard.grasp = json.loads((self.source / 'source.json').read_text())['grasp_command']
        if self.panel:
            self.panel_ready = False
        source_info = json.loads((self.source / 'source.json').read_text())
        self.apply_camera(getattr(self,'evidence_camera',None) or source_info.get('camera', self.camera_state()))
        self.verify_target()
        if self.env.viewer is not None:
            self.env.viewer.update()

    def frame(self, camera=None):
        m, d = self.model_data()
        if self.renderer is None:
            m.vis.global_.offwidth = max(m.vis.global_.offwidth, self.args.width)
            m.vis.global_.offheight = max(m.vis.global_.offheight, self.args.height)
            self.renderer = mujoco.Renderer(m, height=self.args.height, width=self.args.width)
            from OpenGL.GL import glGetString, GL_RENDERER, GL_VERSION
            write_json(self.root / 'renderer.json', {'renderer': glGetString(GL_RENDERER).decode(),
                'version': glGetString(GL_VERSION).decode(), 'width': self.args.width, 'height': self.args.height})
        if camera is not None:
            self.capture_camera.lookat[:] = camera['lookat']
            for attr in ['distance', 'azimuth', 'elevation']:
                setattr(self.capture_camera, attr, camera[attr])
        elif self.env.viewer is not None and self.env.viewer.viewer is not None:
            c = self.env.viewer.viewer.cam
            self.capture_camera.lookat[:] = c.lookat
            for attr in ['distance', 'azimuth', 'elevation']:
                setattr(self.capture_camera, attr, getattr(c, attr))
        if self.recording and self.recording.get('task_video_identity'):
            from mobiwam.visible_object_binding import validate_native_geometry, validate_render_model, frame_binding
            identity=self.recording['task_video_identity']
            validate_native_geometry(m,identity['native']['native_geometry_inventory'])
            validate_render_model(m,self.renderer)
            self.recording['native_frame_binding']=frame_binding(m,d,self.recording['n'],{'lookat':list(self.capture_camera.lookat),'distance':self.capture_camera.distance,'azimuth':self.capture_camera.azimuth,'elevation':self.capture_camera.elevation})
        self.renderer.update_scene(d, camera=self.capture_camera, scene_option=self.render_options)
        frame = self.renderer.render()
        if self.recording and self.recording.get('native_frame_binding'):
            self.recording['native_frame_binding']['raw_rgb_sha256'] = hashlib.sha256(frame.tobytes()).hexdigest()
        return frame

    def begin(self):
        self.restore()
        if getattr(self, 'identity_expected', None):
            from mobiwam.task_video_identity import observe_native
            observe_native(self, self.identity_expected)
        path = self.source / self.route / ('attempt-' + identifier())
        path.mkdir(parents=True)
        identity = None
        if getattr(self, 'identity_expected', None):
            from mobiwam.task_video_identity import recorder_binding
            identity = recorder_binding(self, path, self.identity_expected, self.identity_context)
        h = h5py.File(path / 'demo.hdf5', 'w')
        group = h.create_group('data/demo_0')
        if identity is not None:
            group.attrs['task_video_identity'] = json.dumps(identity)
        group.attrs['model_file'] = (self.source / 'model.xml').read_text()
        group.attrs['env_info'] = json.dumps(self.config)
        group.attrs['ep_meta'] = (self.source / 'ep_meta.json').read_text()
        group.attrs['target_binding'] = (self.root / 'target-binding.json').read_text()
        group.create_dataset('initial_integration', data=self.integration())
        state = self.env.sim.get_state().flatten()
        for name, shape in [('actions', (self.env.action_dim,)), ('states', (len(state),))]:
            group.create_dataset(name, shape=(0,) + shape, maxshape=(None,) + shape, dtype='f8')
        group['states'].resize(1, axis=0)
        group['states'][0] = state
        self.append_observation(group)
        video = imageio.get_writer(path / 'original.mp4', fps=20, codec='libx264',
                                   quality=7, macro_block_size=None)
        self.recording = {'path': path, 'h': h, 'group': group, 'video': video,
                          'trace': (path / 'trace.jsonl').open('w'),
                          'n': 0, 'started': stamp(), 'events': [], 'success_streak': 0}
        self.recording['camera'] = self.camera_state()
        self.recording['task_video_identity'] = identity
        if getattr(self,'panoramic_camera',None):
            self.recording['panoramic_camera']=self.panoramic_camera
            self.recording['panoramic_video']=imageio.get_writer(path/'panoramic.mp4',fps=20,codec='libx264',quality=7,macro_block_size=None)
        self.paused = False
        self.message = 'Recording. Stop saves this attempt, including failures.'
        write_json(path / 'status.json', {'status': 'recording', 'route': self.route,
            'source': str(self.source), 'label': self.label})
        print('RECORDING', self.route, path, flush=True)

    def step(self, action, keys=()):
        if self.recording and self.recording.get('task_video_identity'):
            from mobiwam.visible_object_binding import validate_native_geometry
            m,_=self.model_data()
            validate_native_geometry(m,self.recording['task_video_identity']['native']['native_geometry_inventory'])
        before = self.trace() if self.recording else None
        self.env.step(action)
        if self.recording:
            r = self.recording
            n = r['n']
            g = r['group']
            g['actions'].resize(n + 1, axis=0)
            g['actions'][n] = action
            g['states'].resize(n + 2, axis=0)
            g['states'][n + 1] = self.env.sim.get_state().flatten()
            self.append_observation(g)
            after = self.trace()
            frame = self.frame(r['camera'])
            r['trace'].write(json.dumps({'step': n, 'wall_time': stamp(), 'keys': keys,
                'before': before, 'after': after, 'native_frame_binding': r.get('native_frame_binding'), 'camera': {
                    'lookat': self.capture_camera.lookat.copy(),
                    'distance': self.capture_camera.distance,
                    'azimuth': self.capture_camera.azimuth,
                    'elevation': self.capture_camera.elevation}}, default=lambda x: x.tolist()) + '\n')
            r['video'].append_data(frame)
            if r.get('panoramic_video'):
                # Same native model/data/time, second evidence camera; policy cameras unchanged.
                pano=mujoco.MjvCamera();c=r['panoramic_camera'];pano.lookat[:]=c['lookat']
                pano.distance=c['distance'];pano.azimuth=c['azimuth'];pano.elevation=c['elevation']
                m,d=self.model_data();self.renderer.update_scene(d,camera=pano,scene_option=self.render_options)
                panorama=self.renderer.render().copy();r['panoramic_video'].append_data(panorama)
                if 'panoramic_raw_frame_sha256' not in r:r['panoramic_raw_frame_sha256']=[]
                r['panoramic_raw_frame_sha256'].append(hashlib.sha256(panorama.tobytes()).hexdigest())
                if 'panoramic_frame_bindings' not in r:r['panoramic_frame_bindings']=[]
                from mobiwam.visible_object_binding import frame_binding
                binding=frame_binding(m,d,n,c);binding['raw_rgb_sha256']=r['panoramic_raw_frame_sha256'][-1]
                r['panoramic_frame_bindings'].append(binding)
            r['n'] += 1
            r['success_streak'] = r['success_streak'] + 1 if after['success'] else 0
            if r['n'] % 20 == 0:
                r['h'].flush()
                r['trace'].flush()
            if r['success_streak'] >= 10:
                self.finish('checker_success_10_steps')

    def finish(self, reason):
        if not self.recording:
            return
        r = self.recording
        result = {'status': 'recorded', 'route': self.route, 'source': str(self.source),
                  'started_at': r['started'], 'ended_at': stamp(), 'steps': r['n'],
                  'checker_success': bool(self.env._check_success()), 'reason': reason,
                  'events': r['events'], 'controller_events_schema': 'reference-controller-events-v1',
                  'initial_stow_required': r.get('initial_stow_required',False), 'review_status': 'pending',
                  'collision_and_A_qualification': 'not_verified', 'replay': 'pending',
                  'state_alignment': 'states[0] before actions[0]; states[t+1] after actions[t]',
                  'video_alignment': 'frame[t] after actions[t], 20 simulation Hz; wall clock may be slower'}
        r['group'].attrs['num_samples'] = r['n']
        r['h']['data'].attrs['total'] = r['n']
        r['h']['data'].attrs['env_args'] = json.dumps({'env_name': self.args.task, 'type': 1, 'env_kwargs': self.config})
        identity = r.get('task_video_identity')
        if identity is not None:
            identity['camera'] = r['camera']
            r['group'].attrs['task_video_identity'] = json.dumps(identity)
        r['h'].close()
        r['trace'].close()
        r['video'].close()
        if r.get('panoramic_video'):
            r['panoramic_video'].close()
            write_json(r['path']/'panoramic-binding.json',dict(camera=r['panoramic_camera'],native_model_geometry_sha256=identity['native']['native_geometry_sha256'] if identity else None,
                frames=r['n'],frame_bindings=r['panoramic_frame_bindings'],raw_frame_sha256=r['panoramic_raw_frame_sha256'],sim_time_binding='same native state/time as primary trace after each action',sha256=hashlib.sha256((r['path']/'panoramic.mp4').read_bytes()).hexdigest()))
        if identity is not None:
            from mobiwam.task_video_identity import finalize_recording, human_delivery
            manifest = finalize_recording(r['path'], identity, r['n'])
            result['task_video_manifest'] = str(manifest)
            human_delivery(manifest, r['path'] / 'human-review')
        write_json(r['path'] / 'result.json', result)
        write_json(r['path'] / 'status.json', {'status': 'recorded'})
        self.last_attempt = r['path']
        self.recording = None
        self.paused = True
        self.message = 'Saved. Replay checks actions without human correction.'
        print('SAVED', self.last_attempt, reason, flush=True)

    def identity_preview(self, output, expected, context, camera=None):
        """One native still through recorder binding; zero actions, no route outcome."""
        from mobiwam.task_video_identity import recorder_binding, finalize_recording, human_delivery
        self.restore()
        before = self.integration().copy()
        if camera is not None:
            self.apply_camera(camera)
        output = Path(output).resolve()
        identity = recorder_binding(self, output, expected, context)
        output.mkdir(parents=True, exist_ok=False)
        with h5py.File(output / 'demo.hdf5', 'w') as h:
            g = h.create_group('data/demo_0')
            g.attrs['task_video_identity'] = json.dumps(identity)
            g.attrs['model_file'] = (self.source / 'model.xml').read_text()
            g.attrs['env_info'] = json.dumps(self.config)
            g.create_dataset('actions', shape=(0, self.env.action_dim), dtype='f8')
            g.create_dataset('states', data=[self.env.sim.get_state().flatten()])
            self.append_observation(g)
        (output / 'trace.jsonl').write_text('')
        with imageio.get_writer(output / 'original.mp4', fps=20, codec='libx264', quality=7, macro_block_size=None) as video:
            video.append_data(self.frame(identity['camera']))
        if not np.array_equal(before, self.integration()):
            raise ValueError('zero-action identity preview changed integration state')
        manifest = finalize_recording(output, identity, 0, kind='zero_action_diagnostic')
        human_delivery(manifest, output / 'human-review')
        return manifest

    def replay(self):
        assert self.last_attempt, 'No recorded attempt to replay.'
        path = self.last_attempt / ('replay-' + identifier())
        path.mkdir()
        self.source = self.last_attempt.parent.parent
        self.restore()
        if self.panel:
            self.message = 'REPLAY: live robot input is disabled.'
            self.update_panel('REPLAY')
        with h5py.File(self.last_attempt / 'demo.hdf5') as f:
            actions = f['data/demo_0/actions'][:]
            states = f['data/demo_0/states'][:]
        with (self.last_attempt / 'trace.jsonl').open() as f:
            cameras = [json.loads(line).get('camera') for line in f]
        video = imageio.get_writer(path / 'replay.mp4', fps=20, codec='libx264',
                                   quality=7, macro_block_size=None)
        live_viewer, had_renderer = self.env.viewer, self.env.has_renderer
        if live_viewer:
            live_viewer.close()
        self.env.viewer = None
        self.env.has_renderer = False
        errors = []
        preview = None
        try:
            if self.panel:
                preview = ReplayPreview(self.panel)
            for i, action in enumerate(actions):
                tick = time.monotonic()
                self.env.step(action)
                errors.append(float(np.max(np.abs(self.env.sim.get_state().flatten() - states[i+1]))))
                frame = self.frame(cameras[i])
                video.append_data(frame)
                if preview:
                    preview.show(frame)
                    self.message = f'Replaying {i+1}/{len(actions)}. Robot input disabled.'
                    self.update_panel('REPLAY')
                    self.keyboard.clear()
                    while not self.keyboard.commands.empty():
                        self.keyboard.commands.get_nowait()
                time.sleep(max(0, .05 - (time.monotonic() - tick)))
        finally:
            video.close()
            if preview:
                preview.close()
            self.env.has_renderer = had_renderer
            self.env.viewer = live_viewer
            if live_viewer:
                live_viewer.update()
                self.panel_ready = False
            if self.keyboard:
                self.keyboard.clear()
                while not self.keyboard.commands.empty():
                    self.keyboard.commands.get_nowait()
        write_json(path / 'result.json', {'ended_at': stamp(), 'steps': len(actions),
            'checker_success': bool(self.env._check_success()), 'max_state_abs_error': max(errors, default=None),
            'first_state_error_gt_1e-5': next((i for i,e in enumerate(errors) if e > 1e-5), None),
            'state_errors': errors, 'human_correction': False, 'review_status': 'pending'})
        print('REPLAY SAVED', path, flush=True)
        self.paused = True
        self.message = 'Replay saved. Result still needs contact/cooperation review.'
        return errors

    def create_panel(self):
        import tkinter as tk
        self.panel = tk.Tk()
        self.panel.title('MobiWAM controls')
        self.panel.geometry('400x900+1200+0')
        self.panel.configure(bg='#243447')
        self.panel.protocol('WM_DELETE_WINDOW', lambda: self.keyboard.commands.put('pause'))
        self.status_label = tk.Label(self.panel, fg='white', bg='#243447',
                                     font=('sans', 16), wraplength=380, justify='left')
        self.status_label.pack(padx=12, pady=12)
        for label, cmd in [('Pause / Resume (Esc)', 'pause'),
                           ('1. Save source (F1)', 'f1'), ('2. Record from source (F2)', 'f2'),
                           ('3. Stop and save (F3)', 'f3'), ('Mark contact (F4)', 'f4'),
                           ('Mark docked / lock D base (F5)', 'f5'), ('Select A / E / D (F6)', 'f6'),
                           ('Restore source (F8)', 'f8'), ('Replay last attempt (F9)', 'f9')]:
            tk.Button(self.panel, text=label, font=('sans', 12),
                      command=lambda c=cmd: self.keyboard.commands.put(c)).pack(fill='x', padx=12, pady=3)
        tk.Label(self.panel, text='ARM: arrows, ; / .\nROTATE: O/P, Y/H, E/R\nBASE: W/A/S/D, Z/X\nGRIPPER: Space\n\nBoth groups work together.\nE base locked; D locks after Dock.\n\nHuman references only.\nNo formal OBC training data.',
                 fg='white', bg='#243447', font=('sans', 12), justify='left').pack(padx=12, pady=12)
        self.panel_ready = False

        # This application owns a dedicated X desktop. Route keyboard events
        # away from MuJoCo's built-in shortcuts (W toggles wireframe, etc.).
        # pynput's XRecord listener still receives the events; mouse events
        # remain ungrabbed for camera navigation and the control panel.
        from Xlib import X, display
        self.panel.update()
        self.keyboard_display = display.Display()
        status = self.keyboard_display.screen().root.grab_keyboard(
            False, X.GrabModeAsync, X.GrabModeAsync, X.CurrentTime)
        self.keyboard_display.sync()
        if status != X.GrabSuccess:
            self.keyboard_display.close()
            raise RuntimeError(f'Cannot isolate teleoperation keyboard: X grab status {status}')
        print('KEYBOARD ISOLATED: viewer shortcuts disabled; mouse camera active', flush=True)

    def update_panel(self, mode=None):
        while self.keyboard_display.pending_events():
            self.keyboard_display.next_event()
        mode = mode or ('PAUSED' if self.paused else 'RECORDING' if self.recording else 'PRACTICE')
        steps = self.recording['n'] if self.recording else '-'
        target_status = ''
        fixture = self.env.drawer if self.args.task == 'CloseDrawer' else self.env.door_fxtr
        opening = max(fixture.get_door_state(env=self.env).values())
        target_status = f'\n{self.target_name}: {opening:.1%} open (goal <=5%)'
        self.status_label.config(text=f'{mode} | human {self.route}\nSteps: {steps}{target_status}\n{self.message}')
        self.panel.update()
        if not self.panel_ready and self.env.viewer and self.env.viewer.viewer:
            from Xlib import display
            d = display.Display()
            for w in d.screen().root.query_tree().children:
                if (w.get_wm_name() or '').startswith('MuJoCo'):
                    w.configure(x=0, y=0, width=1200, height=900)
                    self.panel_ready = True
            d.sync()
            d.close()
        self.panel.lift()

    def self_test(self):
        if self.args.task == 'CloseSingleDoor':
            m, d = self.model_data()
            state = self.integration()
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, self.env.door_fxtr.name + '_microjoint')
            checks = []
            try:
                for angle, expected in [(-np.pi/2, False), (0., True)]:
                    d.qpos[m.jnt_qposadr[jid]] = angle
                    mujoco.mj_forward(m, d)
                    observed = bool(self.env._check_success())
                    checks.append({'angle': angle, 'expected': expected, 'observed': observed})
                    assert observed == expected, 'Checker does not follow the bound microwave door'
            finally:
                mujoco.mj_setState(m, d, state, self.kind)
                mujoco.mj_forward(m, d)
            write_json(self.root / 'checker-binding-test.json', {'checks': checks,
                'engineering_only': True, 'initial_state_restored': True})
        self.freeze()
        self.begin()
        before = self.trace()
        actions, states = [], []
        for _ in range(12):
            a = self.robot.create_action_vector(key_actions({'up', 'd'}, False))
            self.step(a)
            actions.append(a.copy())
            states.append(self.env.sim.get_state().flatten().copy())
        after = self.trace()
        self.finish('engineering_self_test')
        splits = self.robot.composite_controller._action_split_indexes
        for part in ['right', 'base']:
            lo, hi = splits[part]
            assert np.any(actions[0][lo:hi]), f'{part} action was dropped'
        errors = self.replay()
        imageio.imwrite(self.root / 'render-check.png', self.frame())
        result = {'status': 'completed', 'before': before, 'after': after,
                  'base_displacement': float(np.linalg.norm(after['base_pos']-before['base_pos'])),
                  'arm_joint_displacement': float(np.linalg.norm(after['arm_qpos']-before['arm_qpos'])),
                  'max_replay_state_abs_error': max(errors), 'action_splits': splits,
                  'steps': len(actions), 'success_reference': False}
        np.savez(self.root / 'self-test.npz', actions=actions, states=states)
        write_json(self.root / 'self-test.json', result)
        assert result['base_displacement'] > .005, 'Base response is too small; check friction and mapping'
        assert result['arm_joint_displacement'] > 1e-5, 'No actual arm movement'
        assert result['arm_joint_displacement'] < 1., 'Excessive arm motion for small inputs'
        assert max(errors) < 1e-5, 'Restore/action replay diverged'
        print('SELF TEST PASSED', result['base_displacement'], max(errors), flush=True)

    def run(self):
        self.create_panel()
        if self.source:
            self.restore()
        print('READY: arrows/;/./o/p/y/h/e/r arm; WASD base; ZX base yaw; Space gripper', flush=True)
        print('F1 source | F2 record | F3 finish | F4 contact | F5 dock | F6 A/E/D | F8 restore | F9 replay', flush=True)
        try:
            while True:
                tick = time.monotonic()
                while not self.keyboard.commands.empty():
                    cmd = self.keyboard.commands.get_nowait()
                    try:
                        if cmd == 'pause':
                            self.paused = not self.paused
                            self.keyboard.clear()
                            if self.recording:
                                self.recording['events'].append({'step': self.recording['n'], 'event': 'pause' if self.paused else 'resume'})
                        elif cmd == 'f3': self.finish('human_stop')
                        elif self.recording:
                            if cmd in ('f4', 'f5'):
                                self.recording['events'].append({'step': self.recording['n'], 'event': 'contact' if cmd=='f4' else 'docked'})
                                if cmd == 'f5': self.docked = True
                        elif cmd == 'f1': self.freeze()
                        elif cmd == 'f2': self.begin()
                        elif cmd == 'f6':
                            self.route = {'A': 'E', 'E': 'D', 'D': 'A'}[self.route]
                            print('ROUTE', self.route, flush=True)
                        elif cmd == 'f8':
                            self.restore()
                            self.panel_ready = False
                        elif cmd == 'f9': self.replay()
                    except (AssertionError, ValueError) as e:
                        self.message = str(e)
                        print('COMMAND:', e, flush=True)
                ac, keys = self.keyboard.read(self.route == 'E' or (self.route == 'D' and self.docked))
                if not self.paused:
                    self.step(self.robot.create_action_vector(ac), keys)
                self.update_panel()
                time.sleep(max(0, .05 - (time.monotonic() - tick)))
        finally:
            self.finish('process_exit')
            self.keyboard.listener.stop()
            from Xlib import X
            self.keyboard_display.ungrab_keyboard(X.CurrentTime)
            self.keyboard_display.close()
            self.panel.destroy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--task', choices=['CloseDrawer', 'CloseSingleDoor'], default='CloseDrawer')
    p.add_argument('--layout', type=int, default=0)
    p.add_argument('--style', type=int, default=0)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--width', type=int, default=1920)
    p.add_argument('--height', type=int, default=1080)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--source', help='Resume practice/collection from an existing source directory')
    p.add_argument('--replay-attempt', help='Replay an existing attempt with no input device')
    p.add_argument('--resume-attempt', help='Resume paused interactive UI with an existing attempt available for F9')
    args = p.parse_args()
    ref = Reference(args)
    try:
        if args.self_test:
            ref.self_test()
        elif args.replay_attempt:
            ref.replay()
        else:
            ref.run()
    finally:
        if ref.observation_renderer is not None: ref.observation_renderer.close()
        if ref.renderer is not None: ref.renderer.close()
        ref.env.close()


if __name__ == '__main__':
    main()
