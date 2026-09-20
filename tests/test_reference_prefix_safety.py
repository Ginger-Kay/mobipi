import numpy as np
import pytest
from mobiwam.reference_prefix_safety import JointMarginMonitor


def observe(monitor, value, when='after_integration'):
    return monitor.observe([value], step=0, substep=0, phase='stow', when=when)


@pytest.mark.parametrize('value', [.015, np.nextafter(.985, 1.), 0., 1., -.01, 1.01])
def test_strict_boundary_and_outside_rejected(value):
    monitor = JointMarginMonitor([0], [[0., 1.]], ['joint'])
    assert observe(monitor, value)['kind'] == 'predicted_joint_margin'


def test_interior_and_minimum_receipt():
    monitor = JointMarginMonitor([0], [[0., 1.]], ['joint'])
    assert observe(monitor, .5, 'initial') is None
    assert observe(monitor, .01501) is None
    assert observe(monitor, .6) is None
    assert monitor.receipt()['samples'] == 3
    assert monitor.minimum['margin_rad'] == .01501


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_fails_closed(value):
    monitor = JointMarginMonitor([0], [[0., 1.]], ['joint'])
    assert observe(monitor, value)['kind'] == 'predicted_nonfinite_joint_state'


def test_saved_microwave_terminal_state_is_rejected():
    monitor = JointMarginMonitor([0], [[-1.7628, 1.7628]], ['robot0_joint2'])
    failure = observe(monitor, -1.7484406743095378)
    assert failure['kind'] == 'predicted_joint_margin'
    assert failure['margin_rad'] == pytest.approx(.014359325690462166)


@pytest.mark.parametrize('limits', [[[0., .02]], [[1., 0.]], [[0., np.inf]]])
def test_invalid_bounds_rejected(limits):
    with pytest.raises(ValueError):
        JointMarginMonitor([0], limits, ['joint'])


@pytest.mark.parametrize('initial,expected_steps,when', [(.01, 0, 'initial'), (.5, 1, 'after_integration')])
def test_preview_stops_without_an_extra_substep(monkeypatch, tmp_path, initial, expected_steps, when):
    import importlib.util
    import sys
    from pathlib import Path
    from types import SimpleNamespace as NS
    counts = dict(steps=0, closed=False)
    data = NS(qpos=np.array([0., 0., 0., initial]), qvel=np.zeros(4),
              site_xpos=np.array([[1., 0., 0.]]), xpos=np.zeros((1, 3)),
              xmat=np.eye(3).reshape(1, 9), site_xmat=np.eye(3).reshape(1, 9), contact=[])
    model = NS(jnt_qposadr=np.arange(4), jnt_range=np.array([[0., 1.]] * 4), ngeom=0)
    def step():
        counts['steps'] += 1
        data.qpos[3] = .01
    env = NS(lite_physics=True, timestep=0, cur_time=0., control_timestep=.05, model_timestep=.002,
             sim=NS(step1=lambda: None, step2=step), _pre_action=lambda *a: None,
             _update_observables=lambda: None, _get_observations=lambda: None,
             close=lambda: counts.update(closed=True))
    class Ref:
        def __init__(self, args):
            self.env=env
            self.robot=NS(part_controllers={'base': NS(qpos_index=[0, 1, 2], qvel_index=[0, 1, 2]),
                                            'right': NS(qpos_index=[3])}, eef_site_id={'right': 0})
            self.base_body='base'
            self.renderer=self.observation_renderer=None
        def restore(self):
            pass
        def model_data(self):
            return model, data
    monkeypatch.setitem(sys.modules, 'teleop_reference', NS(Reference=Ref, stamp=lambda: 'test'))
    monkeypatch.setitem(sys.modules, 'reference_executor', NS(
        mapped_action=lambda *a: (np.zeros(11), .5, .5, .5),
        ProgressWatch=lambda: NS(update=lambda *a: 0)))
    monkeypatch.setitem(sys.modules, 'reference_geometry', NS(PalmClearance=lambda ref: NS(apply=lambda a: (a, []))))
    path = Path(__file__).resolve().parents[1] / 'scripts/reference_prefix_preview.py'
    spec = importlib.util.spec_from_file_location('isolated_prefix_preview', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'mujoco', NS(mj_name2id=lambda *a: 0, mj_id2name=lambda *a: 'joint',
        mjtObj=NS(mjOBJ_BODY=0, mjOBJ_JOINT=1, mjOBJ_GEOM=2)))
    result = module.preview_prefix(NS(args=NS(task='CloseSingleDoor'), source='unused'),
        dict(id=6, dock=[0., 0., 0.]), tmp_path / 'preview', horizon=2)
    assert not result['valid']
    assert result['failure']['kind'] == 'predicted_joint_margin'
    assert result['failure']['when'] == when
    assert counts['steps'] == result['physics_substeps'] == expected_steps
    assert counts['closed']
