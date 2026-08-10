"""Run the formal EC-1 handoff probe on the full navigation boundary.

The probe uses the same ``move_to_pose`` navigation operation as the full
Mobi-π evaluator, then enters the downstream policy rollout without an
environment reset (the ``reset_before_rollout=False`` boundary).  It is a
handoff reproducibility gate, not a task-success benchmark.  The formal
contract records that collision checking is intentionally an open-path
adapter because EC-1 isolates downstream state restoration.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import mimicgen.utils.pose_utils as PoseUtils
import robosuite.utils.transform_utils as T

from mobipi.macros import DATA_ROOT_DIR, POLICY_CKPT_ROOT_DIR
from mobipi.utils.controller_state import env_controller_adapters
from mobipi.utils.env_utils import get_metadata, load_env
from mobipi.utils.handoff_state import (
    capture_handoff_snapshot,
    restore_handoff_snapshot,
)
from mobipi.utils.nav_utils import move_to_pose
from mobipi.utils.policy_utils import get_config_for_policy, load_policy


class _OpenPathCollisionChecker:
    """Collision adapter that declares the deterministic EC-1 path free.

    EC-1 tests the downstream handoff contract, not navigation collision
    planning.  ``move_to_pose`` still executes all waypoint generation and
    environment steps; this adapter removes scene-model / point-cloud state
    from the formal interface probe and is recorded in the run manifest.
    """

    robot_size = 0.0

    def compute_score(self, _rendered, _edited, poses, numpy=False, **_kwargs):
        count = len(poses)
        if numpy:
            return np.ones(count, dtype=np.bool_)
        return torch.ones(count, dtype=torch.bool, device=poses.device)


def _seed_evaluation(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compare_observations(expected, actual):
    if set(expected) != set(actual):
        return {"matched": False, "reason": "observation_keys_changed"}

    max_abs_diff = 0.0
    for key in sorted(expected):
        expected_array = np.asarray(expected[key])
        actual_array = np.asarray(actual[key])
        if expected_array.shape != actual_array.shape:
            return {
                "matched": False,
                "reason": f"observation_shape_changed:{key}",
                "expected_shape": list(expected_array.shape),
                "actual_shape": list(actual_array.shape),
            }
        if expected_array.size:
            max_abs_diff = max(
                max_abs_diff,
                float(np.max(np.abs(expected_array - actual_array))),
            )
    return {
        "matched": bool(max_abs_diff == 0.0),
        "max_abs_diff": max_abs_diff,
    }


def _compare_arrays(expected, actual, tolerance=1e-6):
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    if expected.shape != actual.shape:
        return {
            "matched": False,
            "max_abs_diff": None,
            "expected_shape": list(expected.shape),
            "actual_shape": list(actual.shape),
        }
    max_abs_diff = 0.0 if expected.size == 0 else float(np.max(np.abs(expected - actual)))
    return {
        "matched": bool(np.allclose(expected, actual, atol=tolerance, rtol=0.0)),
        "max_abs_diff": max_abs_diff,
    }


def _state_probe(env):
    state = env.get_state()
    raw_env = env.unwrapped.env
    success = raw_env._check_success()
    if isinstance(success, dict):
        success = {str(key): bool(value) for key, value in success.items()}
    else:
        success = bool(success)
    return {
        "states": np.array(state["states"], copy=True),
        "ncon": int(raw_env.sim.data.ncon),
        "success": success,
    }


def _run_probe(args):
    config, ckpt_path = get_config_for_policy(
        args.ckpt_root_dir,
        args.data_root_dir,
        args.env_name,
        args.policy_name,
        seed=args.policy_seed,
        dataset_name=args.dataset_name,
    )
    _, rollout_model, _ = load_policy(config, ckpt_path)

    env_meta_list, _ = get_metadata(config)
    camera_names = list(env_meta_list[0]["env_kwargs"]["camera_names"])
    if "robot0_navview" not in camera_names:
        camera_names.append("robot0_navview")
    override_args = {
        "layout_and_style_ids": [[args.layout_id, args.style_id]],
        "place_robot_for_nav": True,
        "camera_names": camera_names,
        "hard_reset": True,
    }
    env = load_env(
        config,
        override_args,
        render_offscreen_override=True,
    )
    env.unwrapped.env.place_robot_for_nav_rng = np.random.default_rng(
        args.environment_seed
    )
    env.reset()

    # load_env intentionally seeds its own construction path with zero.  The
    # evaluation seed starts after reset, before the frozen navigation path.
    _seed_evaluation(args.evaluation_seed)
    robot_controller = env.unwrapped.env.robots[0].composite_controller
    initial_pos, initial_rot = robot_controller.get_controller_base_pose("right")
    initial_pos = np.asarray(initial_pos, dtype=np.float64)
    initial_rot = np.asarray(initial_rot, dtype=np.float64)
    target_pos = initial_pos.copy()
    target_pos[:2] += np.asarray(
        [args.navigation_delta_x, args.navigation_delta_y], dtype=np.float64
    )
    target_rot = initial_rot @ T.quat2mat(
        T.axisangle2quat([0.0, 0.0, args.navigation_delta_yaw])
    )
    target_pose = PoseUtils.make_pose(target_pos, target_rot)
    nav_info = move_to_pose(
        env,
        target_pose,
        render=False,
        verbose=False,
        use_rrt=False,
        k_col=_OpenPathCollisionChecker(),
        legacy=True,
    )

    controller_get_state, controller_set_state = env_controller_adapters(env)
    snapshot = capture_handoff_snapshot(
        env,
        controller_get_state=controller_get_state,
        controller_set_state=controller_set_state,
        torch_module=torch,
        numpy_generators={
            "place_robot_for_nav_rng": env.unwrapped.env.place_robot_for_nav_rng
        },
        required_numpy_generators=("place_robot_for_nav_rng",),
        require_controller=True,
        require_torch=True,
        require_cuda=True,
        metadata={
            "evaluator": "mobipi/eval/eval_ec1_handoff.py",
            "probe": "post_navigation_controller_handoff",
            "env_name": args.env_name,
            "layout_id": args.layout_id,
            "style_id": args.style_id,
            "policy_seed": args.policy_seed,
            "environment_seed": args.environment_seed,
            "evaluation_seed": args.evaluation_seed,
            "navigation": "mobipi.utils.nav_utils.move_to_pose",
            "reset_before_rollout": False,
            "collision_check": "open_path_ec1_adapter",
        },
    )
    handoff_observation = copy.deepcopy(env._get_stacked_obs_from_history())
    controller_parts = list(snapshot.controller_state["parts"])

    replicate_results = []
    for replicate in range(args.replicates):
        restore_handoff_snapshot(
            env,
            snapshot,
            controller_set_state=controller_set_state,
            torch_module=torch,
            numpy_generators={
                "place_robot_for_nav_rng": env.unwrapped.env.place_robot_for_nav_rng
            },
            require_controller=True,
            require_torch=True,
            require_cuda=True,
        )
        restored_observation = env._get_stacked_obs_from_history()
        observation_result = _compare_observations(
            handoff_observation, restored_observation
        )

        rollout_model.start_episode(lang=getattr(env, "_ep_lang_str", "dummy"))
        observations = restored_observation
        actions = []
        states = []
        for _ in range(args.probe_steps):
            action = np.asarray(rollout_model(ob=observations))
            actions.append(action.copy())
            observations, _, _, _ = env.step(action)
            states.append(_state_probe(env))

        replicate_results.append(
            {
                "replicate": replicate,
                "observation": observation_result,
                "actions": actions,
                "states": states,
            }
        )

    reference = replicate_results[0]
    for result in replicate_results:
        result["action_comparisons"] = [
            _compare_arrays(reference["actions"][step], action)
            for step, action in enumerate(result["actions"])
        ]
        result["state_comparisons"] = [
            {
                "states": _compare_arrays(
                    reference["states"][step]["states"], state["states"]
                ),
                "ncon_match": reference["states"][step]["ncon"] == state["ncon"],
                "success_match": reference["states"][step]["success"] == state["success"],
            }
            for step, state in enumerate(result["states"])
        ]

    all_observations_match = all(
        result["observation"]["matched"] for result in replicate_results
    )
    all_actions_match = all(
        comparison["matched"]
        for result in replicate_results
        for comparison in result["action_comparisons"]
    )
    all_states_match = all(
        comparison["states"]["matched"]
        and comparison["ncon_match"]
        and comparison["success_match"]
        for result in replicate_results
        for comparison in result["state_comparisons"]
    )

    report = {
        "schema_version": "ec1-probe-v2",
        "run_id": args.run_id,
        "code_commit": args.code_commit,
        "research_commit": args.research_commit,
        "output": str(args.output) if args.output is not None else None,
        "checkpoint": ckpt_path,
        "env_name": args.env_name,
        "layout_id": args.layout_id,
        "style_id": args.style_id,
        "policy_seed": args.policy_seed,
        "environment_seed": args.environment_seed,
        "evaluation_seed": args.evaluation_seed,
        "navigation": {
            "method": "mobipi.utils.nav_utils.move_to_pose",
            "reset_before_rollout": False,
            "collision_check": "open_path_ec1_adapter",
            "legacy": True,
            "delta_xy_m": [
                args.navigation_delta_x,
                args.navigation_delta_y,
            ],
            "delta_yaw_rad": args.navigation_delta_yaw,
            "initial_base_vec": nav_info["base_vec_history"][0],
            "final_base_vec": nav_info["base_vec_history"][-1],
            "env_step_count": len(nav_info["base_cmd_history"]),
        },
        "controller_parts": controller_parts,
        "replicates": args.replicates,
        "probe_steps": args.probe_steps,
        "gates": {
            "observation_restore": all_observations_match,
            "action_restore": all_actions_match,
            "state_restore": all_states_match,
            "ec1_reproducible": all_observations_match
            and all_actions_match
            and all_states_match,
        },
        "replicate_results": replicate_results,
    }
    return report


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default="CloseSingleDoor")
    parser.add_argument("--policy-name", default="bc_xfmr")
    parser.add_argument("--dataset-name", default="mg-300")
    parser.add_argument("--policy-seed", type=int, default=1)
    parser.add_argument("--environment-seed", type=int, default=10000)
    parser.add_argument("--layout-id", type=int, default=1)
    parser.add_argument("--style-id", type=int, default=1)
    parser.add_argument("--evaluation-seed", type=int, default=23)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--probe-steps", type=int, default=5)
    parser.add_argument("--navigation-delta-x", type=float, default=0.1)
    parser.add_argument("--navigation-delta-y", type=float, default=0.0)
    parser.add_argument("--navigation-delta-yaw", type=float, default=0.0)
    parser.add_argument("--ckpt-root-dir", default=POLICY_CKPT_ROOT_DIR)
    parser.add_argument("--data-root-dir", default=DATA_ROOT_DIR)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--research-commit", required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = _run_probe(args)
    serialized = json.dumps(report, indent=2, sort_keys=True, default=_json_default)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    return 0 if report["gates"]["ec1_reproducible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
