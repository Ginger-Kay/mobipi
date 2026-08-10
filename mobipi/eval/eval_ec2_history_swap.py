"""Run the minimal EC-2 FrameStack past-history swap experiment.

This evaluator is intentionally scoped to ``CloseSingleDoor`` with the frozen
``bc_xfmr / vanilla_policy`` interface. It intervenes on FrameStack observation
history, not on policy-internal memory. Every hybrid keeps simulator,
controller, RNG, wrapper timestep, and current frame from its physical source.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import imageio
import mimicgen.utils.pose_utils as PoseUtils
import numpy as np
import torch

from mobipi.eval.eval_ec1_handoff import (
    _OpenPathCollisionChecker,
    _json_default,
    _seed_evaluation,
    _state_probe,
)
from mobipi.macros import DATA_ROOT_DIR, POLICY_CKPT_ROOT_DIR
from mobipi.utils.controller_state import env_controller_adapters
from mobipi.utils.env_utils import get_metadata, load_env
from mobipi.utils.handoff_state import (
    capture_handoff_snapshot,
    restore_handoff_snapshot,
)
from mobipi.utils.history_swap import (
    HISTORY_SWAP_SCHEMA_VERSION,
    build_history_intervention,
    validate_history_intervention,
)
from mobipi.utils.nav_utils import angle_wrap, move_to_pose
from mobipi.utils.policy_utils import get_config_for_policy, load_policy


CELL_ORDER = (("A", "A"), ("A", "B"), ("B", "B"), ("B", "A"))
META_OBSERVATION_KEYS = {"actions", "timesteps"}


def _max_abs(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return None
    if np.issubdtype(left.dtype, np.bool_) or np.issubdtype(right.dtype, np.bool_):
        return 0.0 if np.array_equal(left, right) else 1.0
    return 0.0 if left.size == 0 else float(np.max(np.abs(left - right)))


def _numeric_leaves(value, prefix="root"):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _numeric_leaves(value[key], f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _numeric_leaves(item, f"{prefix}[{index}]")
    else:
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number) or np.issubdtype(
            array.dtype, np.bool_
        ):
            yield prefix, array


def _compare_nested_numeric(left, right):
    left_leaves = dict(_numeric_leaves(left))
    right_leaves = dict(_numeric_leaves(right))
    if set(left_leaves) != set(right_leaves):
        return {"schema_match": False, "max_abs_diff": None}
    maximum = 0.0
    for key in sorted(left_leaves):
        difference = _max_abs(left_leaves[key], right_leaves[key])
        if difference is None:
            return {
                "schema_match": False,
                "max_abs_diff": None,
                "mismatched_leaf": key,
            }
        maximum = max(maximum, difference)
    return {"schema_match": True, "max_abs_diff": maximum}


def _current_frames(snapshot):
    if len(snapshot.frame_stacks) != 1:
        raise RuntimeError("EC-2 requires exactly one FrameStack wrapper")
    stack = snapshot.frame_stacks[0]
    return {
        key: np.asarray(frames[-1]).copy()
        for key, frames in stack.obs_history.items()
    }


def _history_difference(snapshot_a, snapshot_b, threshold):
    stack_a = snapshot_a.frame_stacks[0]
    stack_b = snapshot_b.frame_stacks[0]
    if set(stack_a.obs_history) != set(stack_b.obs_history):
        return {"schema_match": False, "meaningfully_different": False}
    per_key = {}
    meaningful_non_meta = False
    for key in sorted(stack_a.obs_history):
        frames_a = stack_a.obs_history[key][:-1]
        frames_b = stack_b.obs_history[key][:-1]
        differences = [_max_abs(a, b) for a, b in zip(frames_a, frames_b)]
        if any(value is None for value in differences):
            return {
                "schema_match": False,
                "meaningfully_different": False,
                "mismatched_key": key,
            }
        maximum = max(differences, default=0.0)
        per_key[key] = {"past_max_abs_diff": maximum}
        if key not in META_OBSERVATION_KEYS and maximum > threshold:
            meaningful_non_meta = True
    return {
        "schema_match": True,
        "meaningfully_different": meaningful_non_meta,
        "threshold": threshold,
        "per_key": per_key,
    }


def _terminal_match(snapshot_a, snapshot_b, nav_a, nav_b, args):
    final_a = np.asarray(nav_a["segments"][-1]["base_vec_history"][-1])
    final_b = np.asarray(nav_b["segments"][-1]["base_vec_history"][-1])
    xy_diff = float(np.max(np.abs(final_a[:2] - final_b[:2])))
    yaw_diff = float(abs(angle_wrap(float(final_a[2] - final_b[2]))))

    state_diff = _max_abs(
        snapshot_a.env_state["states"], snapshot_b.env_state["states"]
    )
    controller = _compare_nested_numeric(
        snapshot_a.controller_state, snapshot_b.controller_state
    )
    current_a = _current_frames(snapshot_a)
    current_b = _current_frames(snapshot_b)
    if set(current_a) != set(current_b):
        current = {"schema_match": False, "per_key": {}}
    else:
        current = {"schema_match": True, "per_key": {}}
        for key in sorted(current_a):
            left = current_a[key]
            right = current_b[key]
            if left.shape != right.shape:
                current["schema_match"] = False
                current["per_key"][key] = {"shape_match": False}
                continue
            absolute = np.abs(left.astype(np.float64) - right.astype(np.float64))
            current["per_key"][key] = {
                "shape_match": True,
                "max_abs_diff": 0.0 if absolute.size == 0 else float(absolute.max()),
                "mean_abs_diff": 0.0 if absolute.size == 0 else float(absolute.mean()),
                "shape": list(left.shape),
                "dtype": str(left.dtype),
            }

    image_means = [
        value["mean_abs_diff"]
        for key, value in current["per_key"].items()
        if "image" in key and value.get("shape_match")
    ]
    lowdim_maxima = [
        value["max_abs_diff"]
        for key, value in current["per_key"].items()
        if "image" not in key
        and key not in META_OBSERVATION_KEYS
        and value.get("shape_match")
    ]
    raw_a = nav_a["terminal_probe"]
    raw_b = nav_b["terminal_probe"]
    contact_success_match = (
        raw_a["ncon"] == raw_b["ncon"] and raw_a["success"] == raw_b["success"]
    )

    gates = {
        "base_pose": xy_diff <= args.base_xy_tolerance
        and yaw_diff <= args.base_yaw_tolerance,
        "simulator_state": state_diff is not None
        and state_diff <= args.sim_state_tolerance,
        "controller_state": controller["schema_match"]
        and controller["max_abs_diff"] <= args.controller_tolerance,
        "current_observation_schema": current["schema_match"],
        "current_observation_values": (
            max(image_means, default=0.0) <= args.current_image_mean_tolerance
            and max(lowdim_maxima, default=0.0) <= args.current_lowdim_tolerance
        ),
        "contact_success": contact_success_match,
    }
    gates["passed"] = all(gates.values())
    return {
        "thresholds": {
            "base_xy_m": args.base_xy_tolerance,
            "base_yaw_rad": args.base_yaw_tolerance,
            "simulator_state_max_abs": args.sim_state_tolerance,
            "controller_max_abs": args.controller_tolerance,
            "current_image_mean_abs": args.current_image_mean_tolerance,
            "current_lowdim_max_abs": args.current_lowdim_tolerance,
        },
        "observed": {
            "base_xy_max_abs_m": xy_diff,
            "base_yaw_abs_rad": yaw_diff,
            "simulator_state_max_abs": state_diff,
            "controller": controller,
            "current_observation": current,
            "contact_success_match": contact_success_match,
        },
        "gates": gates,
    }


def _pose(initial_pos, initial_rot, delta_x, delta_y):
    position = initial_pos.copy()
    position[:2] += np.asarray([delta_x, delta_y], dtype=np.float64)
    return PoseUtils.make_pose(position, initial_rot)


def _execute_approach(env, poses, label):
    segments = []
    for pose in poses:
        segments.append(
            move_to_pose(
                env,
                pose,
                render=False,
                verbose=False,
                use_rrt=False,
                k_col=_OpenPathCollisionChecker(),
                legacy=True,
            )
        )
    return {
        "label": label,
        "segments": segments,
        "env_step_count": int(
            sum(len(segment["base_cmd_history"]) for segment in segments)
        ),
        "terminal_probe": _state_probe(env),
    }


def _capture(env, controller_get_state, label):
    return capture_handoff_snapshot(
        env,
        controller_get_state=controller_get_state,
        controller_set_state=lambda _state: None,
        torch_module=torch,
        numpy_generators={
            "place_robot_for_nav_rng": env.unwrapped.env.place_robot_for_nav_rng
        },
        required_numpy_generators=("place_robot_for_nav_rng",),
        require_controller=True,
        require_torch=True,
        require_cuda=True,
        metadata={
            "evaluator": "mobipi/eval/eval_ec2_history_swap.py",
            "approach": label,
            "history_semantics": "FrameStack observation history, not policy memory",
        },
    )


def _restore(env, snapshot, controller_set_state):
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


def _current_preserved(observation, physical_snapshot):
    expected = _current_frames(physical_snapshot)
    if set(observation) != set(expected):
        return False
    return all(
        np.array_equal(np.asarray(observation[key])[-1], expected[key][0])
        if expected[key].shape[0] == 1
        else np.array_equal(np.asarray(observation[key])[-1], expected[key])
        for key in expected
    )


def _run_cells(env, rollout_model, snapshots, hybrids, controller_set_state, args):
    records = {f"x_{x}_h_{h}": [] for x, h in CELL_ORDER}
    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=True)
    for replicate in range(args.replicates):
        for physical_source, history_source in CELL_ORDER:
            cell = f"x_{physical_source}_h_{history_source}"
            hybrid = hybrids[(physical_source, history_source)]
            validate_history_intervention(
                hybrid, snapshots[physical_source], snapshots[history_source]
            )
            _restore(env, hybrid, controller_set_state)
            observation = copy.deepcopy(env._get_stacked_obs_from_history())
            if not _current_preserved(observation, snapshots[physical_source]):
                raise RuntimeError(f"current frame preservation failed for {cell}")
            rollout_model.start_episode(lang=getattr(env, "_ep_lang_str", "dummy"))
            actions = []
            states = []
            video_path = (
                args.video_dir / f"{cell}__replicate_0.mp4"
                if args.video_dir is not None and replicate == 0
                else None
            )
            writer = (
                imageio.get_writer(str(video_path), fps=args.video_fps)
                if video_path is not None
                else None
            )
            try:
                if writer is not None:
                    writer.append_data(
                        env.render(mode="rgb_array", height=512, width=512)
                    )
                for _step in range(args.probe_steps):
                    action = np.asarray(rollout_model(ob=observation))
                    actions.append(action.copy())
                    observation, _, _, _ = env.step(action)
                    states.append(_state_probe(env))
                    if writer is not None:
                        writer.append_data(
                            env.render(mode="rgb_array", height=512, width=512)
                        )
            finally:
                if writer is not None:
                    writer.close()
            records[cell].append(
                {
                    "replicate": replicate,
                    "physical_source": physical_source,
                    "history_source": history_source,
                    "current_source": physical_source,
                    "video_uri": str(video_path) if video_path is not None else None,
                    "actions": actions,
                    "states": states,
                }
            )
    return records


def _trajectory_arrays(records, cell):
    actions = np.asarray(
        [[step for step in record["actions"]] for record in records[cell]]
    )
    states = np.asarray(
        [
            [step["states"] for step in record["states"]]
            for record in records[cell]
        ]
    )
    return actions, states


def _repeat_noise(records, cell):
    actions, states = _trajectory_arrays(records, cell)
    action_noise = float(np.max(np.abs(actions - actions[0:1])))
    state_noise = float(np.max(np.abs(states - states[0:1])))
    contact_success_stable = all(
        record["states"][step]["ncon"] == records[cell][0]["states"][step]["ncon"]
        and record["states"][step]["success"]
        == records[cell][0]["states"][step]["success"]
        for record in records[cell]
        for step in range(len(record["states"]))
    )
    return {
        "action_max_abs": action_noise,
        "state_max_abs": state_noise,
        "contact_success_stable": contact_success_stable,
    }


def _effect(records, physical_source):
    cell_a = f"x_{physical_source}_h_A"
    cell_b = f"x_{physical_source}_h_B"
    actions_a, states_a = _trajectory_arrays(records, cell_a)
    actions_b, states_b = _trajectory_arrays(records, cell_b)
    action_delta = actions_b - actions_a
    state_delta = states_b - states_a
    return {
        "action_max_abs": float(np.max(np.abs(action_delta))),
        "state_max_abs": float(np.max(np.abs(state_delta))),
        "action_delta_mean": np.mean(action_delta, axis=0),
    }


def _cosine(left, right):
    left = np.asarray(left).reshape(-1)
    right = np.asarray(right).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return None
    return float(np.dot(left, right) / denominator)


def _classify(records, args):
    noise = {
        "x_A_h_A": _repeat_noise(records, "x_A_h_A"),
        "x_B_h_B": _repeat_noise(records, "x_B_h_B"),
    }
    identities_stable = all(
        item["action_max_abs"] <= args.repeat_tolerance
        and item["state_max_abs"] <= args.repeat_tolerance
        and item["contact_success_stable"]
        for item in noise.values()
    )
    effects = {source: _effect(records, source) for source in ("A", "B")}
    cosine = _cosine(
        effects["A"]["action_delta_mean"], effects["B"]["action_delta_mean"]
    )
    both_below = all(
        item["action_max_abs"] <= args.action_effect_tolerance
        and item["state_max_abs"] <= args.state_effect_tolerance
        for item in effects.values()
    )
    both_action_above = all(
        item["action_max_abs"] > args.action_effect_tolerance
        for item in effects.values()
    )
    if not identities_stable:
        decision = "Inconclusive"
        reason = "identity_controls_unstable"
    elif both_below:
        decision = "No-Go"
        reason = "history_swap_not_above_preregistered_tolerance"
    elif both_action_above and cosine is not None and cosine >= args.effect_cosine_min:
        decision = "Go"
        reason = "repeatable_history_source_consistent_action_divergence"
    else:
        decision = "Inconclusive"
        reason = "mixed_or_directionally_inconsistent_swap_effect"
    return {
        "decision": decision,
        "reason": reason,
        "identity_controls": noise,
        "identity_controls_stable": identities_stable,
        "effects": effects,
        "history_effect_cosine": cosine,
        "thresholds": {
            "repeat_max_abs": args.repeat_tolerance,
            "action_effect_max_abs": args.action_effect_tolerance,
            "state_effect_max_abs": args.state_effect_tolerance,
            "effect_cosine_min": args.effect_cosine_min,
        },
    }


def _approach_definition(args):
    definition = {
        "A": {"waypoints_delta_xy_m": [[args.target_x, args.target_y]]},
        "B": {
            "waypoints_delta_xy_m": [
                [args.detour_x, args.detour_y],
                [args.target_x, args.target_y],
            ]
        },
        "rotation": "initial base rotation for every waypoint",
        "move_to_pose": {"use_rrt": False, "legacy": True, "collision": "open"},
    }
    encoded = json.dumps(definition, sort_keys=True, separators=(",", ":")).encode()
    return definition, hashlib.sha256(encoded).hexdigest()


def _run(args):
    if args.env_name != "CloseSingleDoor" or args.policy_name != "bc_xfmr":
        raise RuntimeError("formal EC-2 is frozen to CloseSingleDoor / bc_xfmr")
    config, checkpoint = get_config_for_policy(
        args.ckpt_root_dir,
        args.data_root_dir,
        args.env_name,
        args.policy_name,
        seed=args.policy_seed,
        dataset_name=args.dataset_name,
    )
    if int(config.train.frame_stack) != 10:
        raise RuntimeError("formal EC-2 requires FrameStack length 10")
    _, rollout_model, _ = load_policy(config, checkpoint)
    env_meta_list, _ = get_metadata(config)
    camera_names = list(env_meta_list[0]["env_kwargs"]["camera_names"])
    if "robot0_navview" not in camera_names:
        camera_names.append("robot0_navview")
    env = load_env(
        config,
        {
            "layout_and_style_ids": [[args.layout_id, args.style_id]],
            "place_robot_for_nav": True,
            "camera_names": camera_names,
            "hard_reset": True,
        },
        render_offscreen_override=True,
    )
    env.unwrapped.env.place_robot_for_nav_rng = np.random.default_rng(
        args.environment_seed
    )
    env.reset()
    _seed_evaluation(args.evaluation_seed)
    controller_get_state, controller_set_state = env_controller_adapters(env)
    initial = _capture(env, controller_get_state, "initial")
    controller = env.unwrapped.env.robots[0].composite_controller
    initial_pos, initial_rot = controller.get_controller_base_pose("right")
    initial_pos = np.asarray(initial_pos, dtype=np.float64)
    initial_rot = np.asarray(initial_rot, dtype=np.float64)
    target = _pose(initial_pos, initial_rot, args.target_x, args.target_y)
    detour = _pose(initial_pos, initial_rot, args.detour_x, args.detour_y)

    _restore(env, initial, controller_set_state)
    nav_a = _execute_approach(env, [target], "A_direct")
    snapshot_a = _capture(env, controller_get_state, "A")
    _restore(env, initial, controller_set_state)
    nav_b = _execute_approach(env, [detour, target], "B_detour")
    snapshot_b = _capture(env, controller_get_state, "B")
    snapshots = {"A": snapshot_a, "B": snapshot_b}

    history_difference = _history_difference(
        snapshot_a, snapshot_b, args.history_difference_tolerance
    )
    terminal = _terminal_match(snapshot_a, snapshot_b, nav_a, nav_b, args)
    identifiable = (
        history_difference["schema_match"]
        and history_difference["meaningfully_different"]
        and terminal["gates"]["passed"]
    )
    approach_definition, approach_sha256 = _approach_definition(args)
    report = {
        "schema_version": "ec2-history-swap-result-v1",
        "run_id": args.run_id,
        "code_commit": args.code_commit,
        "research_commit": args.research_commit,
        "checkpoint": checkpoint,
        "environment": {
            "task": args.env_name,
            "layout_id": args.layout_id,
            "style_id": args.style_id,
            "policy_seed": args.policy_seed,
            "environment_seed": args.environment_seed,
            "evaluation_seed": args.evaluation_seed,
        },
        "intervention": {
            "schema_version": HISTORY_SWAP_SCHEMA_VERSION,
            "past_indices": list(range(9)),
            "current_index": 9,
            "current_source": "physical_source",
            "policy_internal_memory_claimed": False,
        },
        "approach_definition": approach_definition,
        "approach_definition_sha256": approach_sha256,
        "approaches": {"A": nav_a, "B": nav_b},
        "history_difference": history_difference,
        "terminal_match": terminal,
        "gates": {"identifiable": identifiable},
        "preflight_only": bool(args.preflight_only),
    }
    if not identifiable or args.preflight_only:
        report["decision"] = "preflight_pass" if identifiable else "Inconclusive"
        report["decision_reason"] = (
            "terminal_and_history_gates_passed"
            if identifiable
            else "terminal_or_history_identifiability_gate_failed"
        )
        return report, None

    hybrids = {
        (physical, history): build_history_intervention(
            snapshots[physical],
            snapshots[history],
            physical_source=physical,
            history_source=history,
        )
        for physical, history in CELL_ORDER
    }
    records = _run_cells(
        env, rollout_model, snapshots, hybrids, controller_set_state, args
    )
    classification = _classify(records, args)
    report["replicates"] = args.replicates
    report["probe_steps"] = args.probe_steps
    report["cell_order_per_replicate"] = [
        f"x_{physical}_h_{history}" for physical, history in CELL_ORDER
    ]
    report["cells"] = records
    report["videos"] = [
        records[cell][0]["video_uri"]
        for cell in records
        if records[cell][0]["video_uri"] is not None
    ]
    report["classification"] = classification
    report["decision"] = classification["decision"]
    report["decision_reason"] = classification["reason"]

    arrays = {}
    for cell in records:
        arrays[f"{cell}__actions"], arrays[f"{cell}__states"] = _trajectory_arrays(
            records, cell
        )
    return report, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default="CloseSingleDoor")
    parser.add_argument("--policy-name", default="bc_xfmr")
    parser.add_argument("--dataset-name", default="mg-300")
    parser.add_argument("--policy-seed", type=int, default=1)
    parser.add_argument("--environment-seed", type=int, default=10000)
    parser.add_argument("--evaluation-seed", type=int, default=23)
    parser.add_argument("--layout-id", type=int, default=1)
    parser.add_argument("--style-id", type=int, default=1)
    parser.add_argument("--target-x", type=float, default=0.10)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--detour-x", type=float, default=0.0)
    parser.add_argument("--detour-y", type=float, default=0.08)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--probe-steps", type=int, default=5)
    parser.add_argument("--base-xy-tolerance", type=float, default=0.005)
    parser.add_argument("--base-yaw-tolerance", type=float, default=0.01)
    parser.add_argument("--sim-state-tolerance", type=float, default=0.02)
    parser.add_argument("--controller-tolerance", type=float, default=0.02)
    parser.add_argument("--current-image-mean-tolerance", type=float, default=0.02)
    parser.add_argument("--current-lowdim-tolerance", type=float, default=0.02)
    parser.add_argument("--history-difference-tolerance", type=float, default=1e-4)
    parser.add_argument("--repeat-tolerance", type=float, default=1e-6)
    parser.add_argument("--action-effect-tolerance", type=float, default=1e-4)
    parser.add_argument("--state-effect-tolerance", type=float, default=1e-5)
    parser.add_argument("--effect-cosine-min", type=float, default=0.5)
    parser.add_argument("--ckpt-root-dir", default=POLICY_CKPT_ROOT_DIR)
    parser.add_argument("--data-root-dir", default=DATA_ROOT_DIR)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--research-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npz-output", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    report, arrays = _run(args)
    serialized = json.dumps(report, indent=2, sort_keys=True, default=_json_default)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n")
    if arrays is not None:
        args.npz_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.npz_output, **arrays)
    return 0 if report["decision"] in {"preflight_pass", "Go", "No-Go"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
