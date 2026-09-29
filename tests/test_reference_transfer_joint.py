"""DR-v0.2 drawer target transfer preserves both physical anchors."""
import unittest

import numpy as np

from mobiwam.reference_transfer import (
    _drawer_opening_from_joint,
    _map_drawer_joint_by_closed_progress,
)


class DrawerJointTransferTest(unittest.TestCase):
    def setUp(self):
        self.old_range = np.array([-0.6, 0.0])
        self.new_range = np.array([-0.6, 0.0])
        self.old_initial = -0.32167279257405434
        self.new_initial = -0.30559291760605267

    def map(self, old_q):
        return _map_drawer_joint_by_closed_progress(
            old_q, self.old_initial, self.new_initial,
            self.old_range, self.new_range)

    def test_initial_and_closed_endpoint_match_new_model(self):
        self.assertAlmostEqual(self.map(self.old_initial), self.new_initial)
        self.assertAlmostEqual(self.map(0.0), 0.0)
        self.assertAlmostEqual(self.map(self.old_initial / 2), self.new_initial / 2)

    def test_old_final_waypoint_stays_within_new_native_limit(self):
        old_final = 0.001355613923972654 + self.old_initial - self.new_initial
        self.assertLessEqual(self.map(old_final), self.new_range[1])
        self.assertGreater(self.map(old_final), self.new_initial)

    def test_out_of_range_reference_is_rejected_without_clipping(self):
        for old_q in (0.00001, -0.61):
            with self.subTest(old_q=old_q), self.assertRaises(ValueError):
                self.map(old_q)

    def test_out_of_range_mapped_opening_is_rejected(self):
        with self.assertRaises(ValueError):
            _map_drawer_joint_by_closed_progress(
                -0.59, self.old_initial, self.new_initial,
                self.old_range, np.array([-0.5, 0.0]))

    def test_invalid_new_source_opening_is_rejected(self):
        with self.assertRaises(ValueError):
            _map_drawer_joint_by_closed_progress(
                self.old_initial, self.old_initial, 0.001,
                self.old_range, self.new_range)

    def test_checker_opening_uses_mapped_joint_not_old_delta(self):
        extent = -self.new_initial / 0.9260391442607656
        mapped = self.map(-0.01472421125608)
        opening = _drawer_opening_from_joint(mapped, extent)
        self.assertAlmostEqual(_drawer_opening_from_joint(self.new_initial, extent),
                               0.9260391442607656)
        self.assertAlmostEqual(_drawer_opening_from_joint(0.0, extent), 0.0)
        self.assertGreater(opening, 0.03)
        self.assertLess(opening, 0.06)

    def test_opening_or_closed_endpoint_semantics_must_match_native_drawer(self):
        with self.assertRaises(ValueError):
            _drawer_opening_from_joint(-0.4, 0.3)
        with self.assertRaises(ValueError):
            _map_drawer_joint_by_closed_progress(
                self.old_initial, self.old_initial, self.new_initial,
                self.old_range, np.array([-0.6, 0.01]))


if __name__ == '__main__':
    unittest.main()
