"""Run reproducible paired candidate rollouts for a frozen policy checkpoint."""

from __future__ import annotations

import os
import random
import sys
import traceback
from pathlib import Path

import click
import imageio
import numpy as np
import torch

from mobipi.utils.env_utils import get_metadata, load_env
from mobipi.utils.door_diagnostics import (
    DoorDiagnosticRecorder,
    load_case_list,
    validate_case_list,
)
from mobipi.utils.paired_rollout_utils import (
    artifact_record,
    base_manifest,
    candidate_run_id,
    collect_reset_fingerprint,
    compare_payloads,
    environment_record,
    load_candidate_config,
    pose_error,
    repo_state,
    requested_pose,
    resolve_seeds,
    robot_collision_pairs,
    sanitized_command,
    scene_group_id,
    sha256_file,
    target_relative_pose,
    utc_now,
    validate_candidate,
    validate_manifest,
    verify_config_sha256,
    workspace_boundary_check,
    write_json,
)
from mobipi.utils.policy_utils import get_config_for_policy, load_policy


def _close_env(env):
    current = env
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "close"):
            try:
                current.close()
                return
            except Exception:
                pass
        if hasattr(current, "env"):
            current = current.env
        else:
            return


def _make_env(
    config,
    env_metadata,
    layout_id,
    style_id,
    environment_seed,
    force_pose=None,
    cpu_reset_only=False,
):
    override_args = {
        "layout_and_style_ids": [[layout_id, style_id]],
        "camera_names": list(env_metadata["env_kwargs"]["camera_names"]),
        "seed": int(environment_seed),
        "hard_reset": True,
        "randomize_base_init_pose": None,
    }
    if force_pose is not None:
        override_args["force_robot_placement"] = (
            np.array([force_pose[0], force_pose[1], 0.0], dtype=np.float64),
            np.array([0.0, 0.0, force_pose[2]], dtype=np.float64),
        )
    env = load_env(
        config,
        override_args,
        use_image_obs_override=False if cpu_reset_only else None,
        render_offscreen_override=False if cpu_reset_only else None,
    )
    observation = env.reset()
    return env, observation


def _latest_observation(observation):
    result = {}
    for key, value in observation.items():
        array = np.asarray(value)
        result[key] = array[-1].copy() if array.ndim and array.shape[0] == 10 else array.copy()
    return result


def _run_policy_episode(
    policy,
    env,
    initial_observation,
    horizon,
    terminate_on_success,
    video_path,
    video_skip,
    diagnostic_recorder=None,
):
    observation = initial_observation
    policy.start_episode(lang=env._ep_lang_str)
    actions = []
    observations = [_latest_observation(observation)]
    success = False
    termination_reason = "horizon"
    writer = imageio.get_writer(video_path, fps=20) if video_path else None
    try:
        for step in range(horizon):
            action = policy(ob=observation)
            actions.append(np.asarray(action).copy())
            observation, _, _, info = env.step(action)
            if diagnostic_recorder is not None:
                # Read-only instrumentation after the existing step.  It does
                # not issue another step or alter the policy/action path.
                diagnostic_recorder.record_state(action=action)
            observations.append(_latest_observation(observation))
            success = bool(info["is_success"]["task"])
            if writer is not None and step % video_skip == 0:
                writer.append_data(env.render(mode="rgb_array", height=512, width=512))
            if terminate_on_success and success:
                termination_reason = "success"
                break
    finally:
        if writer is not None:
            writer.close()
    trajectory = {"actions": np.stack(actions) if actions else np.empty((0,))}
    for key in observations[0]:
        trajectory[f"obs_latest__{key}"] = np.stack([item[key] for item in observations])
    if diagnostic_recorder is None:
        return trajectory, success, len(actions), termination_reason, None
    diagnostic_trajectory, diagnostic_summary, diagnostic_contacts = diagnostic_recorder.finalize()
    trajectory.update(diagnostic_trajectory)
    return (
        trajectory,
        success,
        len(actions),
        termination_reason,
        (diagnostic_summary, diagnostic_contacts),
    )


@click.command()
@click.option("--experiment_id", default="mobipi-paired-candidate-rollout-d4-6")
@click.option("--run_id", required=True)
@click.option("--research_repo", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--research_commit", required=True)
@click.option("--env_name", default="CloseSingleDoor")
@click.option("--policy_name", default="bc_xfmr")
@click.option("--baseline_name", default="vanilla_policy")
@click.option("--checkpoint_seed", default=None, type=int)
@click.option("--seed", "legacy_seed", default=None, type=int, hidden=True)
@click.option("--environment_seeds", required=True, help="Comma-separated environment seeds")
@click.option("--candidate_seed", default=0, type=int)
@click.option("--evaluation_seed", default=0, type=int)
@click.option("--layout_id", default=1, type=int)
@click.option("--style_id", default=1, type=int)
@click.option("--candidate_config", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--candidate_config_sha256", required=True)
@click.option("--candidate_ids", default=None, help="Optional comma-separated candidate subset")
@click.option(
    "--diagnostic_case_list",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Explicit diagnostic cases; prevents environment x candidate Cartesian expansion",
)
@click.option("--diagnostic_case_list_sha256", default=None)
@click.option("--shuffle_candidates/--no_shuffle_candidates", default=False)
@click.option("--ckpt_root_dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--data_root_dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output_root", required=True, type=click.Path(file_okay=False))
@click.option("--horizon", default=500, type=int)
@click.option("--terminate_on_success/--no_terminate_on_success", default=True)
@click.option("--video_skip", default=5, type=int)
@click.option("--smoke_only", is_flag=True, help="Create/reset/fingerprint only; do not run policy actions")
@click.option("--load_policy_in_smoke", is_flag=True)
def main(
    experiment_id,
    run_id,
    research_repo,
    research_commit,
    env_name,
    policy_name,
    baseline_name,
    checkpoint_seed,
    legacy_seed,
    environment_seeds,
    candidate_seed,
    evaluation_seed,
    layout_id,
    style_id,
    candidate_config,
    candidate_config_sha256,
    candidate_ids,
    diagnostic_case_list,
    diagnostic_case_list_sha256,
    shuffle_candidates,
    ckpt_root_dir,
    data_root_dir,
    output_root,
    horizon,
    terminate_on_success,
    video_skip,
    smoke_only,
    load_policy_in_smoke,
):
    if baseline_name != "vanilla_policy":
        raise click.ClickException("paired candidate evaluator currently supports vanilla_policy only")
    seeds, warning = resolve_seeds(
        checkpoint_seed, legacy_seed, 0, candidate_seed, evaluation_seed
    )
    if warning:
        click.echo(f"WARNING: {warning}", err=True)
    parsed_environment_seeds = [int(item.strip()) for item in environment_seeds.split(",") if item.strip()]
    if not parsed_environment_seeds:
        raise click.ClickException("--environment_seeds must not be empty")
    candidate_config_data = load_candidate_config(candidate_config)
    if not smoke_only and candidate_config_data["review_status"] != "frozen":
        raise click.ClickException(
            "full rollout requires a researcher-reviewed candidate config with review_status=frozen"
        )
    config_sha256 = verify_config_sha256(candidate_config, candidate_config_sha256)
    candidates = list(candidate_config_data["candidates"])
    for candidate in candidates:
        validate_candidate(candidate, candidate_config_data)
    diagnostic_cases = None
    diagnostic_cases_by_key = {}
    diagnostic_cases_by_environment = {}
    diagnostic_case_list_checksum = None
    if diagnostic_case_list:
        if candidate_ids:
            raise click.ClickException("--candidate_ids cannot be combined with --diagnostic_case_list")
        if shuffle_candidates:
            raise click.ClickException("diagnostic case order is frozen; do not shuffle candidates")
        if not diagnostic_case_list_sha256:
            raise click.ClickException("--diagnostic_case_list_sha256 is required with --diagnostic_case_list")
        diagnostic_case_list_checksum = sha256_file(diagnostic_case_list)
        if diagnostic_case_list_checksum.lower() != diagnostic_case_list_sha256.lower():
            raise click.ClickException(
                "diagnostic case-list SHA-256 mismatch: "
                f"expected {diagnostic_case_list_sha256}, got {diagnostic_case_list_checksum}"
            )
        diagnostic_cases = validate_case_list(
            load_case_list(diagnostic_case_list),
            candidate_ids=[candidate["candidate_id"] for candidate in candidates],
            candidate_config_sha256=config_sha256,
        )
        for case in diagnostic_cases["cases"]:
            key = (int(case["environment_seed"]), str(case["candidate_id"]))
            if key in diagnostic_cases_by_key:
                raise click.ClickException(f"duplicate diagnostic environment/candidate case: {key}")
            diagnostic_cases_by_key[key] = case
            diagnostic_cases_by_environment.setdefault(int(case["environment_seed"]), []).append(
                str(case["candidate_id"])
            )
        if set(parsed_environment_seeds) != set(diagnostic_cases_by_environment):
            raise click.ClickException(
                "--environment_seeds must exactly match environments in --diagnostic_case_list"
            )
    elif candidate_ids:
        selected = set(item.strip() for item in candidate_ids.split(",") if item.strip())
        candidates = [candidate for candidate in candidates if candidate["candidate_id"] in selected]
        if {candidate["candidate_id"] for candidate in candidates} != selected:
            raise click.ClickException("--candidate_ids contains unknown candidate")
    if shuffle_candidates:
        random.Random(candidate_seed).shuffle(candidates)

    config, ckpt_path = get_config_for_policy(
        ckpt_root_dir, data_root_dir, env_name, policy_name, seed=seeds["checkpoint_seed"], dataset_name="mg-300"
    )
    checkpoint_sha256 = sha256_file(ckpt_path)
    policy = None
    if not smoke_only or load_policy_in_smoke:
        _, policy, _ = load_policy(config, ckpt_path)
    env_meta = get_metadata(config)[0][0]
    code_state = repo_state(Path(__file__).resolve().parents[2])
    research_state = repo_state(research_repo)
    if research_state["commit"] != research_commit:
        raise click.ClickException("--research_commit does not match research repo HEAD")
    command, _ = sanitized_command(sys.argv)
    parent_output = Path(output_root).resolve() / run_id
    parent_output.mkdir(parents=True, exist_ok=False)
    batch_results = []

    for environment_seed in parsed_environment_seeds:
        seeds["environment_seed"] = environment_seed
        probe_env = None
        try:
            probe_env, probe_obs = _make_env(
                config,
                env_meta,
                layout_id,
                style_id,
                environment_seed,
                cpu_reset_only=smoke_only,
            )
            probe_fingerprint = collect_reset_fingerprint(
                probe_env,
                probe_obs,
                env_name,
                layout_id,
                style_id,
                environment_seed,
                tolerance=float(candidate_config_data["fingerprint_tolerance"]),
            )
            nominal_pose = np.asarray(probe_fingerprint["raw"]["base_pose_world_xy_yaw"])
        finally:
            if probe_env is not None:
                _close_env(probe_env)

        group_id = scene_group_id(
            run_id, env_name, layout_id, style_id, seeds["checkpoint_seed"], environment_seed
        )
        expected_invariant_hash = probe_fingerprint["scene_invariant_hash"]
        environment_candidates = candidates
        if diagnostic_cases is not None:
            selected_ids = set(diagnostic_cases_by_environment.get(environment_seed, []))
            environment_candidates = [
                candidate for candidate in candidates if candidate["candidate_id"] in selected_ids
            ]
            if {candidate["candidate_id"] for candidate in environment_candidates} != selected_ids:
                raise click.ClickException(
                    f"diagnostic case list has unknown candidate for environment {environment_seed}"
                )
        for candidate in environment_candidates:
            candidate_id = candidate["candidate_id"]
            case = (
                diagnostic_cases_by_key.get((environment_seed, candidate_id))
                if diagnostic_cases is not None
                else None
            )
            episode_evaluation_seed = int(case["evaluation_seed"]) if case is not None else int(evaluation_seed)
            episode_seeds = {**seeds, "environment_seed": int(environment_seed), "evaluation_seed": episode_evaluation_seed}
            candidate_output = parent_output / group_id / candidate_id
            candidate_output.mkdir(parents=True, exist_ok=False)
            frame = candidate.get("coordinate_frame", candidate_config_data["coordinate_frame"])
            requested = requested_pose(nominal_pose, candidate["requested_transform"], frame)
            candidate_specific_run_id = candidate_run_id(run_id, environment_seed, candidate_id)
            env = None
            actual = None
            fingerprint = None
            manifest_path = candidate_output / "manifest.json"
            execution_log_path = candidate_output / "execution_log.json"
            manifest = base_manifest(
                experiment_id=experiment_id,
                run_id=candidate_specific_run_id,
                scene_id=group_id,
                candidate={
                    "candidate_id": candidate_id,
                    "coordinate_frame": frame,
                    "units": candidate_config_data["units"],
                    "nominal_base_pose": nominal_pose.tolist(),
                    "requested_transform": candidate["requested_transform"],
                    "requested_base_pose": requested.tolist(),
                    "actual_base_pose": None,
                    "target_relative_pose": None,
                    "reset_fingerprint_uri": str((candidate_output / "reset_fingerprint.json").resolve()),
                },
                research={**research_state, "commit": research_commit},
                code=code_state,
                environment=environment_record(torch),
                protocol={
                    "simulator": "Mobi-pi bundled RoboCasa 0.2.0",
                    "task": env_name,
                    "layout_id": layout_id,
                    "style_id": style_id,
                    "policy": f"{policy_name}/{baseline_name}",
                    "checkpoint_uri": str(Path(ckpt_path).resolve()),
                    "checkpoint_sha256": checkpoint_sha256,
                    "horizon": horizon,
                    "terminate_on_success": terminate_on_success,
                    "config_uri": str(Path(candidate_config).resolve()),
                    "config_sha256": config_sha256,
                },
                seeds={"training_seed": int(config.train.seed), **episode_seeds},
                command=command,
                output_root=str(candidate_output.resolve()),
            )
            if diagnostic_cases is not None:
                manifest["diagnostic_case"] = {
                    "schema_version": diagnostic_cases["schema_version"],
                    "case_id": case["case_id"],
                    "expected_historical_outcome": case["expected_historical_outcome"],
                    "matched_pair_id": case["matched_pair_id"],
                    "source_run_id": case.get("source_run_id", "TBD"),
                    "case_list_uri": str(Path(diagnostic_case_list).resolve()),
                    "case_list_sha256": diagnostic_case_list_checksum,
                }
                manifest["protocol"]["diagnostic_case_list_uri"] = str(Path(diagnostic_case_list).resolve())
                manifest["protocol"]["diagnostic_case_list_sha256"] = diagnostic_case_list_checksum
                manifest["protocol"]["diagnostic_schema_version"] = "1.0"
                manifest["protocol"]["diagnostic_config"] = {
                    "schema_version": diagnostic_cases.get("schema_version", "1.0"),
                    "stall_config": diagnostic_cases.get("stall_config", {}),
                    "door_geometry": diagnostic_cases.get("door_geometry", {}),
                }
            manifest["execution"]["log_uri"] = str(execution_log_path.resolve())
            manifest["started_at"] = utc_now()
            manifest["status"] = "running"
            write_json(manifest_path, manifest)
            try:
                env, observation = _make_env(
                    config,
                    env_meta,
                    layout_id,
                    style_id,
                    environment_seed,
                    force_pose=requested,
                    cpu_reset_only=smoke_only,
                )
                fingerprint = collect_reset_fingerprint(
                    env,
                    observation,
                    env_name,
                    layout_id,
                    style_id,
                    environment_seed,
                    tolerance=float(candidate_config_data["fingerprint_tolerance"]),
                )
                actual = fingerprint["raw"]["base_pose_world_xy_yaw"]
                error = pose_error(requested, actual)
                tolerance = float(candidate_config_data["actual_pose_tolerance"])
                if max(abs(item) for item in error["delta"]) > tolerance:
                    raise RuntimeError(f"requested/actual pose error exceeds tolerance {tolerance}: {error}")
                invariant_comparison = compare_payloads(
                    probe_fingerprint["raw_scene_invariant"],
                    fingerprint["raw_scene_invariant"],
                    float(candidate_config_data["paired_invariant_tolerance"]),
                )
                if not invariant_comparison["matched"]:
                    raise RuntimeError(
                        "paired scene invariant mismatch: "
                        f"expected hash {expected_invariant_hash}, got "
                        f"{fingerprint['scene_invariant_hash']}; comparison={invariant_comparison}"
                    )
                collisions = robot_collision_pairs(env)
                if collisions:
                    raise RuntimeError(f"candidate has initial non-floor robot collision: {collisions}")
                boundary_check = workspace_boundary_check(env, actual)
                if (
                    not boundary_check["inside_floor"]
                    or boundary_check["max_fixture_overlap_area_m2"] > 1e-8
                ):
                    raise RuntimeError(f"candidate failed workspace boundary check: {boundary_check}")
                fingerprint_path = candidate_output / "reset_fingerprint.json"
                write_json(fingerprint_path, fingerprint)
                manifest["candidate"]["actual_base_pose"] = actual
                manifest["candidate"]["target_relative_pose"] = target_relative_pose(env, actual)
                manifest["candidate"]["pose_error"] = error
                manifest["candidate"]["boundary_check"] = boundary_check
                manifest["candidate"]["scene_invariant_comparison"] = invariant_comparison
                manifest["artifacts"].append(artifact_record(fingerprint_path, "json", "readable"))

                if smoke_only:
                    success = None
                    episode_length = 0
                    termination_reason = "smoke_only"
                else:
                    random.seed(episode_evaluation_seed)
                    np.random.seed(episode_evaluation_seed)
                    torch.manual_seed(episode_evaluation_seed)
                    diagnostic_recorder = (
                        DoorDiagnosticRecorder(env, stall_config=diagnostic_cases.get("stall_config"))
                        if diagnostic_cases is not None
                        else None
                    )
                    trajectory_path = candidate_output / "trajectory.npz"
                    video_path = candidate_output / "rollout.mp4"
                    trajectory, success, episode_length, termination_reason, diagnostic_result = _run_policy_episode(
                        policy,
                        env,
                        observation,
                        horizon,
                        terminate_on_success,
                        str(video_path),
                        video_skip,
                        diagnostic_recorder=diagnostic_recorder,
                    )
                    np.savez_compressed(trajectory_path, **trajectory)
                    with np.load(trajectory_path) as archive:
                        if "actions" not in archive:
                            raise RuntimeError("trajectory validation failed: actions missing")
                    manifest["artifacts"].append(artifact_record(trajectory_path, "npz", "readable"))
                    manifest["artifacts"].append(artifact_record(video_path, "mp4", "written"))
                    if diagnostic_result is not None:
                        diagnostic_summary, diagnostic_contacts = diagnostic_result
                        diagnostic_summary_path = candidate_output / "diagnostic_summary.json"
                        diagnostic_contacts_path = candidate_output / "diagnostic_contacts.json"
                        write_json(diagnostic_summary_path, diagnostic_summary)
                        write_json(diagnostic_contacts_path, diagnostic_contacts)
                        manifest["diagnostic"] = diagnostic_summary
                        manifest["artifacts"].append(
                            artifact_record(diagnostic_summary_path, "json", "readable")
                        )
                        manifest["artifacts"].append(
                            artifact_record(diagnostic_contacts_path, "json", "readable")
                        )
                        manifest["notes"]["diagnostic_fields"] = diagnostic_summary["field_schema"]
                result_path = candidate_output / "result.json"
                write_json(
                    result_path,
                    {
                        "success": success,
                        "episode_length": episode_length,
                        "termination_reason": termination_reason,
                        "requested_pose": requested.tolist(),
                        "actual_pose": actual,
                        "pose_error": error,
                        "scene_invariant_hash": fingerprint["scene_invariant_hash"],
                        "scene_invariant_comparison": invariant_comparison,
                        "diagnostic_case": manifest.get("diagnostic_case"),
                        "diagnostic": manifest.get("diagnostic"),
                    },
                )
                manifest["artifacts"].append(artifact_record(result_path, "json", "readable"))
                manifest["status"] = "smoke_completed" if smoke_only else "completed"
                manifest["execution"]["exit_code"] = 0
                manifest["events"].append(
                    {
                        "timestamp": utc_now(),
                        "actor": "eval_paired_candidates.py",
                        "event": "candidate completed",
                        "status_before": "running",
                        "status_after": manifest["status"],
                        "note": termination_reason,
                    }
                )
            except Exception as error:
                manifest["status"] = "failed"
                manifest["execution"]["exit_code"] = 1
                manifest["execution"]["failure_reason"] = f"{type(error).__name__}: {error}"
                manifest["events"].append(
                    {
                        "timestamp": utc_now(),
                        "actor": "eval_paired_candidates.py",
                        "event": "candidate failed",
                        "status_before": "running",
                        "status_after": "failed",
                        "note": traceback.format_exc(),
                    }
                )
                raise
            finally:
                if env is not None:
                    _close_env(env)
                manifest["ended_at"] = utc_now()
                write_json(
                    execution_log_path,
                    {
                        "run_id": candidate_specific_run_id,
                        "status": manifest["status"],
                        "started_at": manifest["started_at"],
                        "ended_at": manifest["ended_at"],
                        "exit_code": manifest["execution"]["exit_code"],
                        "failure_reason": manifest["execution"]["failure_reason"],
                        "events": manifest["events"],
                    },
                )
                manifest["artifacts"].append(
                    artifact_record(execution_log_path, "json", "readable")
                )
                validate_manifest(manifest)
                write_json(manifest_path, manifest)
            batch_results.append(
                {
                    "run_id": candidate_specific_run_id,
                    "scene_group_id": group_id,
                    "candidate_id": candidate_id,
                    "manifest_uri": str(manifest_path.resolve()),
                    "status": manifest["status"],
                }
            )

    write_json(
        parent_output / "batch_manifest.json",
        {
            "schema_version": "1.0",
            "experiment_id": experiment_id,
            "run_id": run_id,
            "created_at": utc_now(),
            "config_uri": str(Path(candidate_config).resolve()),
            "config_sha256": config_sha256,
            "diagnostic_case_list_uri": str(Path(diagnostic_case_list).resolve()) if diagnostic_case_list else None,
            "diagnostic_case_list_sha256": diagnostic_case_list_checksum,
            "diagnostic_schema_version": "1.0" if diagnostic_cases is not None else None,
            "results": batch_results,
        },
    )


if __name__ == "__main__":
    main()
