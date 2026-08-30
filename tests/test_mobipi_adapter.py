import unittest

import numpy as np

from mobiwam.adapters.mobipi import (
    MobiPiPairedAdapter,
    PLANAR_BASE_JOINT_NAMES,
    SourceStateIneligibleError,
    SourceStratum,
    _apply_planar_base_lock,
    _capture_planar_base_lock,
    _default_task_root_pose,
    _detour_pose_from_start,
    _is_mobile_base_geom,
    _offset_planar_pose_local,
    _state_hash,
    select_source_stratum,
)


class MobiPiAdapterSamplingTest(unittest.TestCase):
    def test_capture_rejects_initial_mobile_base_collision(self):
        adapter = object.__new__(MobiPiPairedAdapter)
        adapter.env = object()
        adapter.stratum = SourceStratum(1, 1, 0.03, 0)
        adapter.environment_seed = 16
        adapter._base_collision = lambda: True

        with self.assertRaisesRegex(
            SourceStateIneligibleError,
            "environment_seed=16",
        ):
            adapter.capture_source_state()

    def test_planar_base_lock_restores_qpos_and_zeros_qvel(self):
        class FakeModel:
            qpos = {name: index + 1 for index, name in enumerate(PLANAR_BASE_JOINT_NAMES)}
            qvel = {name: index + 5 for index, name in enumerate(PLANAR_BASE_JOINT_NAMES)}

            def get_joint_qpos_addr(self, name):
                return self.qpos[name]

            def get_joint_qvel_addr(self, name):
                return self.qvel[name]

        class FakeData:
            qpos = np.arange(10, dtype=np.float64)
            qvel = np.arange(10, dtype=np.float64)

        class FakeSim:
            model = FakeModel()
            data = FakeData()
            forward_calls = 0

            def forward(self):
                self.forward_calls += 1

        class FakeEnvironment:
            sim = FakeSim()

        environment = FakeEnvironment()
        base_lock = _capture_planar_base_lock(environment)
        expected = base_lock.qpos_values.copy()
        environment.sim.data.qpos[base_lock.qpos_indices] += 1.0
        environment.sim.data.qvel[base_lock.qvel_indices] = 3.0
        _apply_planar_base_lock(environment, base_lock)
        np.testing.assert_allclose(
            environment.sim.data.qpos[base_lock.qpos_indices], expected
        )
        np.testing.assert_allclose(
            environment.sim.data.qvel[base_lock.qvel_indices], 0.0
        )
        self.assertEqual(environment.sim.forward_calls, 1)

    def test_one_hundred_sources_cover_five_layouts_and_four_noise_levels(self):
        strata = [
            select_source_stratum(
                index,
                layouts=[1, 4, 7, 8, 9],
                noise_sigmas=[0.0, 0.03, 0.05, 0.10],
                states_per_noise_per_layout=5,
            )
            for index in range(100)
        ]
        self.assertEqual({item.layout_id for item in strata}, {1, 4, 7, 8, 9})
        for layout in [1, 4, 7, 8, 9]:
            for sigma in [0.0, 0.03, 0.05, 0.10]:
                count = sum(
                    item.layout_id == layout and item.base_noise_sigma == sigma
                    for item in strata
                )
                self.assertEqual(count, 5)

    def test_default_dock_pose_comes_from_task_geometry_not_rng_replay(self):
        class FakeEnvironment:
            door_fxtr = object()

            def compute_robot_base_placement_pose(self, *, ref_fixture):
                self.ref_fixture = ref_fixture
                return np.array([1.0, -2.0, 0.1]), np.array([0.0, 0.0, 0.5])

        environment = FakeEnvironment()
        pose = _default_task_root_pose(environment)
        self.assertIs(environment.ref_fixture, environment.door_fxtr)
        np.testing.assert_allclose(pose[:3, 3], [1.0, -2.0, 0.1])
        self.assertAlmostEqual(np.arctan2(pose[1, 0], pose[0, 0]), 0.5)

    def test_mobile_base_geom_detection_matches_mobipi_naming(self):
        self.assertTrue(_is_mobile_base_geom("mobilebase0_base_collision"))
        self.assertTrue(_is_mobile_base_geom("robot0_mobilebase0_wheel"))
        self.assertFalse(_is_mobile_base_geom("robot0_link7_collision"))
        self.assertFalse(_is_mobile_base_geom(None))

    def test_dock_candidate_offset_is_expressed_in_dock_frame(self):
        dock = np.eye(4)
        dock[:2, :2] = [[0.0, -1.0], [1.0, 0.0]]
        dock[:2, 3] = [1.0, 2.0]
        shifted = _offset_planar_pose_local(dock, [0.0, 0.03])

        np.testing.assert_allclose(shifted[:2, 3], [0.97, 2.0])
        np.testing.assert_allclose(shifted[:3, :3], dock[:3, :3])

    def test_standoff_candidate_combines_longitudinal_and_lateral_offsets(self):
        dock = np.eye(4)
        dock[:2, :2] = [[0.0, -1.0], [1.0, 0.0]]
        dock[:2, 3] = [1.0, 2.0]

        shifted = _offset_planar_pose_local(dock, [0.08, 0.12])

        np.testing.assert_allclose(shifted[:2, 3], [0.88, 2.08])
        np.testing.assert_allclose(shifted[:3, :3], dock[:3, :3])

    def test_detour_delta_is_relative_to_start_in_dock_frame(self):
        dock = np.eye(4)
        dock[:2, :2] = [[0.0, -1.0], [1.0, 0.0]]
        dock[:2, 3] = [1.0, 2.0]
        start = _offset_planar_pose_local(dock, [0.18, -0.10])
        start[:3, :3] = np.eye(3)

        target = _detour_pose_from_start(
            dock,
            start,
            [0.10, 0.12],
            preserve_start_yaw=True,
        )

        expected = _offset_planar_pose_local(dock, [0.28, 0.02])
        np.testing.assert_allclose(target[:2, 3], expected[:2, 3])
        np.testing.assert_allclose(target[:3, :3], start[:3, :3])

    def test_state_hash_ignores_robocasa_object_cfg_rewrite(self):
        before = {
            "model": "<mujoco/>",
            "states": np.array([1.0, 2.0]),
            "ep_meta": '{"lang":"close the door","layout_id":1,'
            '"object_cfgs":[{"info":{"mjcf_path":"old"}}]}',
        }
        after = {
            "model": "<mujoco/>",
            "states": np.array([1.0, 2.0]),
            "ep_meta": '{"object_cfgs":[{"info":{"mjcf_path":"rewritten"}}],'
            '"layout_id":1,"lang":"close the door"}',
        }
        self.assertEqual(_state_hash(before), _state_hash(after))

    def test_state_hash_keeps_task_critical_metadata(self):
        baseline = {
            "model": "<mujoco/>",
            "states": np.array([1.0, 2.0]),
            "ep_meta": '{"lang":"close the door","layout_id":1}',
        }
        changed_language = {
            **baseline,
            "ep_meta": '{"lang":"open the door","layout_id":1}',
        }
        changed_layout = {
            **baseline,
            "ep_meta": '{"lang":"close the door","layout_id":4}',
        }
        self.assertNotEqual(_state_hash(baseline), _state_hash(changed_language))
        self.assertNotEqual(_state_hash(baseline), _state_hash(changed_layout))


if __name__ == "__main__":
    unittest.main()
