import numpy as np

from mobiwam.planner_min import (
    continuous_path_metrics,
    occupancy_lattice_astar,
    rank_primary,
    task_space_region_chain,
    velocity_level_qp,
)


def test_task_space_chains_use_articulation_geometry():
    drawer = task_space_region_chain("CloseDrawer", [1, 0, 1], [0, 0, 1], [1, 0, 0], -0.4, 0.0)
    door = task_space_region_chain("CloseSingleDoor", [1, 0, 1], [0, 0, 1], [0, 0, 1], 0.8, 0.0)
    assert drawer.joint_type == "prismatic"
    assert door.joint_type == "hinge"
    assert not np.allclose(drawer.manipulation_points_world[0], drawer.manipulation_points_world[-1])
    assert not np.allclose(door.manipulation_points_world[0], door.manipulation_points_world[-1])


def test_holonomic_astar_routes_around_obstacle_deterministically():
    free = lambda point: not (0.4 <= point[0] <= 0.6 and -0.2 <= point[1] <= 0.2)
    first = occupancy_lattice_astar([0, 0], [1, 0], free, bounds_xy=[-0.1, 1.1, -0.5, 0.5], resolution_m=0.05)
    second = occupancy_lattice_astar([0, 0], [1, 0], free, bounds_xy=[-0.1, 1.1, -0.5, 0.5], resolution_m=0.05)
    np.testing.assert_array_equal(first, second)
    assert np.max(np.abs(first[:, 1])) > 0.2


def test_continuous_metrics_densify_between_knots():
    result = continuous_path_metrics([[0, 0], [0.2, 0]], lambda point: 0.1 - abs(point[0] - 0.1),
                                     spacing_m=0.01, velocity_limit=1.0, acceleration_limit=100.0, dt_s=0.05)
    assert result["sample_count"] >= 21
    assert result["minimum_continuous_clearance_m"] >= -1e-12


def test_velocity_qp_and_primary_ranking():
    jacobian = np.eye(3)
    velocity, receipt = velocity_level_qp(jacobian, [1, 0, 0], [-0.5] * 3, [0.5] * 3,
                                          base_weight=2.0, damping=1e-3)
    assert receipt["feasible"]
    assert velocity[0] <= 0.5
    rows = [
        {"candidate_id": "d1", "hard_valid": True, "minimum_continuous_clearance_m": .1,
         "minimum_manipulability_or_joint_margin": .2, "minimum_policy_view_compatibility": .3,
         "total_planned_base_path_m": .5, "total_planned_time_s": 4},
        {"candidate_id": "d2", "hard_valid": True, "minimum_continuous_clearance_m": .2,
         "minimum_manipulability_or_joint_margin": .1, "minimum_policy_view_compatibility": .2,
         "total_planned_base_path_m": .8, "total_planned_time_s": 5},
    ]
    assert rank_primary(rows)["candidate_id"] == "d2"
