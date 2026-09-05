import unittest

import numpy as np
from shapely.geometry import Polygon

from mobiwam.scene004 import (
    CANDIDATE_FEATURE_FIELDS,
    FixtureFunctionalRecord,
    a0_command,
    assist_candidates,
    bounded_x_fallback,
    build_minimal_input,
    candidate_feature_vector,
    best_fixed_route,
    camera_grid,
    fixture_anchored_lattice,
    functional_fixture_predicates,
    geometry_rule_select,
    make_minimal_obc,
    minimal_obc_loss,
    prediction_first_select,
    reject_outcome_fields,
    search_lattice,
    select_cell_camera,
    signed_corridor_clearance,
    summarize_test_row,
    trainable_parameter_count,
    validate_functional_fixture,
)


def fixture(task, klass, names, types):
    return FixtureFunctionalRecord(task, "target", klass, True, tuple(names), tuple(types),
                                   tuple((0.0, 1.0) for _ in names), tuple(0.5 for _ in names),
                                   (0.1, 0.2, 0.8), True)


def pose(x=0.0, y=0.0):
    value = np.eye(4); value[:2, 3] = [x, y]; return value


class Scene004Test(unittest.TestCase):
    def test_functional_fixture_accepts_task_native_subtypes(self):
        for row in (
            fixture("CloseSingleDoor", "x.Microwave", ["microjoint"], ["hinge"]),
            fixture("CloseSingleDoor", "x.SingleCabinet", ["doorhinge"], ["hinge"]),
            fixture("CloseDrawer", "x.Drawer", ["slidejoint"], ["slide"]),
        ):
            self.assertTrue(validate_functional_fixture(row)["passed"])
        self.assertFalse(validate_functional_fixture(fixture("CloseSingleDoor", "x.DoubleCabinet", ["l", "r"], ["hinge", "hinge"]))["passed"])
        self.assertFalse(validate_functional_fixture(fixture("CloseSingleDoor", "x.Panel", [], []))["passed"])

    def test_lattice_is_concrete_and_used_by_search(self):
        lattice = fixture_anchored_lattice([0, 0], [0, -1], 0.0)
        self.assertEqual(len(lattice), 27)
        self.assertEqual(len({(p.x_m, p.y_m, p.yaw_rad) for p in lattice}), 27)
        selected, rows = search_lattice(lattice, lambda p: {
            "hard_valid": True, "visibility": 1.0, "reachability": 1.0,
            "joint_margin": 1.0, "intent_error": abs(p.x_m - 0.4), "planned_path": abs(p.y_m),
        })
        self.assertAlmostEqual(selected.x_m, 0.4)
        self.assertEqual(len(rows), 27)

    def test_single_five_cm_inflation_accepts_zero_signed_clearance(self):
        floor = Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)])
        obstacle = Polygon([(0.5, -0.2), (0.8, -0.2), (0.8, 0.2), (0.5, 0.2)])
        # radius + inflation = 0.30, point x=0.20 touches but does not add another 5cm.
        result = signed_corridor_clearance([[0.2, 0.0], [0.2, 0.1]], [("cab", obstacle)], floor,
                                           base_radius_m=0.25, inflation_m=0.05)
        self.assertGreaterEqual(result["min_signed_clearance_m"], -1e-12)
        self.assertTrue(result["passed"])
        self.assertEqual(result["acceptance_threshold_m"], 0.0)

    def test_camera_is_cell_frozen_and_deterministic(self):
        evaluations = {pose.camera_id: [{"passed": True, "min_border_fraction": 0.1,
                                         "base_projected_diameter_px": 130.0}] * 3 for pose in camera_grid()}
        first = select_cell_camera("CloseDrawer-l1", evaluations)
        second = select_cell_camera("CloseDrawer-l1", evaluations)
        self.assertEqual(first["camera_hash"], second["camera_hash"])
        self.assertEqual(first["selected"]["pose"]["fov_deg"], 48.0)
        self.assertEqual(first["selected"]["pose"]["height_m"], 3.2)

    def test_reasons_are_independent_and_class_is_not_a_predicate(self):
        row = fixture("CloseSingleDoor", "x.Microwave", ["microjoint"], ["hinge"])
        preds = functional_fixture_predicates(row)
        self.assertNotIn("fixture_class", preds)
        broken = FixtureFunctionalRecord(**{**row.__dict__, "handle_position_world": None, "task_checker_binding": False})
        result = validate_functional_fixture(broken)
        self.assertEqual(result["failure_reasons"], ["handle_geometry_readable", "task_checker_binding"])

    def test_a_uses_explicit_ee_intent_not_future_base_dimensions(self):
        candidates = assist_candidates([0, 0], pose(), pose(1, 0), [1, 0])
        self.assertEqual([row.candidate_id for row in candidates], ["a1", "a2", "a3", "a4", "a5"])
        self.assertAlmostEqual(candidates[1].planned_net_m, 0.4)
        self.assertEqual(candidates[1].chunks, 4)

    def test_a0_x_and_outcome_guard(self):
        action = np.arange(12, dtype=float)
        expected = action.copy(); expected[7:10] = 0
        np.testing.assert_array_equal(a0_command(action), expected)
        self.assertEqual(bounded_x_fallback()["command"], [0.0] * 12)
        with self.assertRaises(ValueError): reject_outcome_fields({"nested": {"success": True}})

    def test_minimal_model_exact_shape_count_and_loss(self):
        import torch
        model = make_minimal_obc()
        self.assertEqual(trainable_parameter_count(model), 5230)
        candidate = candidate_feature_vector({name: 0.0 for name in CANDIDATE_FEATURE_FIELDS})
        features = build_minimal_input(np.zeros((4, 1024)), candidate)
        self.assertEqual(features.shape, (1045,))
        raw = model(torch.from_numpy(features)[None])
        self.assertEqual(tuple(raw.shape), (1, 5))
        targets = {"success": torch.ones(1), "progress": torch.ones(1), "failure": torch.zeros(1),
                   "base_path_m": torch.zeros(1), "completion_time_s": torch.ones(1)}
        self.assertTrue(torch.isfinite(minimal_obc_loss(raw, targets, timeout_s=20.0)["loss"]))

    def test_frozen_selectors(self):
        rows = [
            {"candidate_id": "e0", "route_family": "E", "stage_eligible": True, "hard_valid": True,
             "predicted_success": .80, "predicted_failure": .20, "predicted_progress": .70,
             "predicted_base_path_m": 0.0, "predicted_completion_time_s": 10,
             "minimum_continuous_clearance_m": .08, "minimum_manipulability_or_joint_margin": .8,
             "minimum_policy_view_compatibility": .8, "total_planned_base_path_m": 0.0,
             "total_planned_time_s": 10.0},
            {"candidate_id": "d1", "route_family": "D", "stage_eligible": True, "hard_valid": True,
             "predicted_success": .86, "predicted_failure": .19, "predicted_progress": .80,
             "predicted_base_path_m": .4, "predicted_completion_time_s": 12,
             "minimum_continuous_clearance_m": .09, "minimum_manipulability_or_joint_margin": .9,
             "minimum_policy_view_compatibility": .9, "total_planned_base_path_m": .4,
             "total_planned_time_s": 12.0},
        ]
        self.assertEqual(prediction_first_select(rows), "d1")
        self.assertEqual(geometry_rule_select(rows), "d1")
        self.assertEqual(prediction_first_select([{**rows[0], "hard_valid": False}]), "X")

    def test_fixed_train_and_all_source_test_denominator(self):
        train = []
        for source in range(3):
            for route, success in (("E", True), ("D", source == 0), ("A", False)):
                train.append({"route_family": route, "success": success, "irreversible_or_collision": False,
                              "progress": float(success), "actual_base_path_m": {"E": 0, "D": .4, "A": .4}[route],
                              "completion_time_s": 10})
        self.assertEqual(best_fixed_route(train), "E")
        test = [
            {"task": "CloseSingleDoor", "success": True, "irreversible_or_collision": False,
             "actual_base_path_m": 0.0, "completion_time_s": 10, "hard_valid": True, "route_family": "E"},
            {"task": "CloseDrawer", "success": False, "irreversible_or_collision": False,
             "actual_base_path_m": 0.0, "completion_time_s": .2, "hard_valid": False, "route_family": "X"},
        ]
        summary = summarize_test_row(test, source_count=2)
        self.assertEqual(summary["overall_success"], 1)
        self.assertEqual(summary["hard_invalid_or_x_count"], 1)


if __name__ == "__main__":
    unittest.main()
