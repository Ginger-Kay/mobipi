import unittest

import numpy as np

from mobiwam.b0_scene_compiler import (
    camera_projection_metrics, continuous_corridor, dock_target, fixture_derived_dock,
    sample_segment, source_lattice, stable_seed,
    validate_a_geometry, validate_d_geometry, validate_fixture, validate_native_frame,
)


class SceneCompilerTest(unittest.TestCase):
    def test_stable_finite_lattice(self):
        self.assertEqual(source_lattice((1, 1), 7), source_lattice((1, 1), 7))
        self.assertEqual(len(source_lattice((1, 1), 7)), 27)
        self.assertEqual(stable_seed("a", 1), stable_seed("a", 1))

    def test_geometry_guards(self):
        dock = dock_target([1, 2, 0], [1, 1, 1])
        self.assertGreaterEqual(validate_d_geometry([1, 1.5], dock), .30)
        validate_a_geometry(.4, 3, True)
        with self.assertRaises(ValueError): validate_d_geometry(dock, dock)
        with self.assertRaises(ValueError): validate_a_geometry(.2, 3, True)

    def test_fixture_and_corridor_guards(self):
        validate_fixture("CloseDrawer", {"class": "robocasa.models.fixtures.cabinets.Drawer", "joint_names": ["slidejoint"], "bbox_size_m": [.5, .5, .5]})
        with self.assertRaises(ValueError): validate_fixture("CloseSingleDoor", {"class": "x.Microwave", "joint_names": ["hinge"]})
        data = continuous_corridor([[0, 0], [.25, 0], [.5, 0]], [.1, .1, .1], spacing_m=.02)
        self.assertEqual(data["length_m"], .5)
        validate_native_frame(np.zeros((1080, 1920, 3), dtype=np.uint8))

    def test_segment_dock_and_projection(self):
        segment = sample_segment([0, 0], [.5, 0])
        self.assertGreaterEqual(len(segment), 26)
        self.assertTrue(np.allclose(segment[[0, -1]], [[0, 0], [.5, 0]]))
        dock = fixture_derived_dock([0, 0, 0], [1, .5, .8], [0, -1], .25)
        self.assertLess(dock[1], 0)
        projection = camera_projection_metrics(
            np.array([[0., 0., 1.], [.1, .1, 1.]]), np.eye(4), width=2, height=2,
            border_fraction=0.0,
        )
        self.assertTrue(projection["passed"])


if __name__ == "__main__":
    unittest.main()
