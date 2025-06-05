"""
Script for evaluating vanilla policy, LeLaN, and BC w/ Nav baselines.

Usage:
python eval_baseline.py --env_name CloseSingleDoor --layout_id -1 --seed 1 \
    --baseline_name vanilla_policy --randomize_base_init_pose 0.00
python eval_baseline.py --env_name CloseSingleDoor --layout_id -1 --seed 1 \
    --baseline_name vanilla_policy --randomize_base_init_pose 0.05
python eval_baseline.py --env_name CloseSingleDoor --layout_id -1 --seed 1 \
    --baseline_name vanilla_policy --randomize_base_init_pose 0.10
python eval_baseline.py --env_name CloseSingleDoor --layout_id -1 --seed 1 \
    --baseline_name il_nav --horizon 800
python eval_baseline.py --env_name CloseSingleDoor --layout_id -1 --seed 1 \
    --baseline_name lelan --check_collisions
python eval_baseline.py --env_name CloseSingleDoor --layout_id -1 --seed 1 \
    --baseline_name lelan_pa --check_collisions --lelan_ckpt_path [path]

@yjy0625
"""
import os

import re
import cv2
import sys
import json
import copy
import h5py
import torch
import click
from glob import glob
from datetime import datetime
import numpy as np
from tqdm import tqdm

from scipy.spatial.transform import Rotation as R

from robosuite.utils.mjmod import CameraModder
from robosuite.utils.camera_utils import get_real_depth_map
import robomimic.utils.train_utils as TrainUtils

from mobipi.nav.lelan import LeLaN

from mobipi.macros import *
from mobipi.utils.constants import *
from mobipi.utils.io_utils import DualStream, camel_to_snake_case
from mobipi.utils.env_utils import get_metadata, load_env, get_env_map_and_default_robot_init_pose
from mobipi.utils.media_utils import save_video, to_fisheye
from mobipi.utils.policy_utils import get_config_for_policy, load_policy
from mobipi.utils.score_utils import PCVacancyDistribution


def compute_robot_angle_from_mat(mat):
    # Compute the yaw (angle around Z-axis) from the quaternion
    ang = np.arctan2(-mat[0, 1], mat[0, 0])
    return ang


def add_text_to_image(image, a1, a2):
    """
    Adds formatted text "(a1, a2)" to the bottom-left corner of an image.
    Chooses text color based on pixel intensity.
    """
    # Convert image to grayscale for intensity analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define bottom-left region to analyze brightness
    h, w = gray.shape
    region_size = 50  # Small patch for brightness check
    bottom_left_patch = gray[h-region_size:h, 0:region_size]

    # Compute average brightness
    avg_intensity = np.mean(bottom_left_patch)

    # Choose text color based on intensity
    text_color = (255, 255, 255) if avg_intensity < 128 else (0, 0, 0)  # White if dark, Black if bright

    # Format the text
    text = f"({a1:.3f}, {a2:.3f})"

    # Define font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    position = (10, h - 10)  # Bottom-left corner with padding

    # Add text to the image
    cv2.putText(image, text, position, font, font_scale, text_color, thickness, cv2.LINE_AA)

    return image


def run_lelan_in_env(env, lelan, init_obs, instruction, action_scale=10.0, num_steps=500,
                     camera_name="robot0_fisheye", collision_checker=None):
    nav_images = []

    base_pos_history = []
    base_mat_history = []
    base_velp_history = []
    base_velr_history = []
    base_cmd_history = []

    robot_base_name = "mobilebase0_base"

    unwrapped_env = env.unwrapped.env
    reset_joint_pos = unwrapped_env.sim.data.qpos[unwrapped_env.robots[0]._ref_joint_pos_indexes]
    obs = init_obs

    for _ in tqdm(range(num_steps)):
        if "fisheye" in camera_name:
            undistorted_image = unwrapped_env.sim.render(width=1024, height=1024,
                                                         camera_name=camera_name)[::-1]
            image = to_fisheye(undistorted_image)
            nav_images.append(np.concatenate([image, undistorted_image], axis=1))
        else:
            images = obs[f"{camera_name}_image"]
            if len(images.shape) == 4:
                image = obs[f"{camera_name}_image"][-1][::-1]
            else:
                image = obs[f"{camera_name}_image"][::-1]
            nav_images.append(image)
        lelan_action = lelan.forward(cv2.resize(image, (224, 224)), instruction)
        action = np.zeros([12])
        action[7] = np.clip(lelan_action.linear.x * action_scale, -1., 1.)
        action[9] = np.clip(lelan_action.angular.z * action_scale, -1., 1.)
        obs, _, _, _ = env.step(action)  # Perform the action
        nav_images[-1] = add_text_to_image(nav_images[-1], action[7], action[9])

        # Set the joint positions for the arm
        unwrapped_env.sim.data.qpos[unwrapped_env.robots[0]._ref_joint_pos_indexes] = reset_joint_pos
        unwrapped_env.sim.forward()

        # Update base history
        base_pos = unwrapped_env.sim.data.get_body_xpos(robot_base_name)
        base_mat = unwrapped_env.sim.data.get_body_xmat(robot_base_name).reshape(3, 3)
        base_pos_history.append(base_pos.copy())
        base_mat_history.append(base_mat.copy())

        base_velp = unwrapped_env.sim.data.get_body_xvelp(robot_base_name)
        base_velr = unwrapped_env.sim.data.get_body_xvelr(robot_base_name)
        base_cmd = np.array([lelan_action.linear.x, 0, lelan_action.angular.z])
        base_velp_history.append(base_velp.copy())
        base_velr_history.append(base_velr.copy())
        base_cmd_history.append(base_cmd.copy())

        # optionally check collisions
        if collision_checker is not None:
            # compute poses
            heading = R.from_matrix(base_mat).as_euler("xyz")[-1]
            poses = np.array([base_pos[0], base_pos[1], heading])[None]

            # outputs 1 for vacant and 0 for collision
            is_collision_free = collision_checker.compute_score(None, None, poses, numpy=True,
                                                                collision_threshold_multiplier=1.0)[0]

            if not is_collision_free:
                print(f"Terminated navigation because of collision at step {_}")
                break

    info = dict(
        base_pos_history=base_pos_history,
        base_mat_history=base_mat_history,
        base_velp_history=base_velp_history,
        base_velr_history=base_velr_history,
        base_cmd_history=base_cmd_history,
        images=nav_images
    )

    return info


def wrap_heading(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


@click.command()
@click.option("--env_name", default="TurnOnSinkFaucet", type=str)
@click.option("--policy_name", default="bc_xfmr", type=str)
@click.option("--seed", default=1, type=int)
@click.option("--layout_id", default=1, type=int)
@click.option("--style_id", default=1, type=int)
@click.option("--ckpt_root_dir", default=POLICY_CKPT_ROOT_DIR, type=str)
@click.option("--data_root_dir", default=DATA_ROOT_DIR, type=str)
@click.option("--log_root_dir", default=LOG_ROOT_DIR, type=str)
@click.option("--num_episodes", default=10, type=int)
@click.option("--baseline_name", default="lelan", type=str,
              help="Choose from [vanilla_policy, lelan, il_nav]")
@click.option("--horizon", default=None, type=int)
@click.option("--randomize_base_init_pose", default=None, type=float)

@click.option("--lelan_ckpt_path", default=None, type=str)

@click.option("--check_collisions", is_flag=True)
@click.option("--scene_model_root_dir", default=SCENE_MODEL_ROOT_DIR, type=str)
@click.option("--scene_model_ckpt_idx", default=29999, type=int)

def main(env_name, policy_name, seed, layout_id, style_id, ckpt_root_dir, data_root_dir,
         log_root_dir, num_episodes, baseline_name, horizon,
         randomize_base_init_pose, lelan_ckpt_path, check_collisions, scene_model_root_dir,
         scene_model_ckpt_idx):
    log_parent_dir = os.path.join(log_root_dir, env_name,
                                  baseline_name + ("" if randomize_base_init_pose is None else f"_dev{randomize_base_init_pose:.2f}"),
                                  policy_name)

    # Create a timestamped log file
    os.makedirs(log_parent_dir, exist_ok=True)
    log_file = os.path.join(
        log_parent_dir, f"console_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_stream = open(log_file, "a")

    # Replace stdout and stderr
    sys.stdout = DualStream(file_stream, sys.stdout)
    sys.stderr = DualStream(file_stream, sys.stderr)

    config, ckpt_path = get_config_for_policy(
        ckpt_root_dir, data_root_dir, env_name,
        policy_name + ("_nav" if baseline_name == "il_nav" else ""), seed=seed,
        dataset_name=("mg-1500-nav" if baseline_name == "il_nav" else "mg-300")
    )
    model, rollout_model, lang_encoder = load_policy(config, ckpt_path)

    # initialize nav policy
    if baseline_name.startswith("lelan"):
        lelan = LeLaN(lelan_ckpt_path)

    if layout_id >= 0:
        layout_and_style_ids = [[layout_id, style_id]]
    else:
        layout_and_style_ids = [[i, i] for i in TEST_LAYOUTS]

    for layout_style_id in layout_and_style_ids:
        layout_id, style_id = layout_style_id
        log_dir = os.path.join(log_parent_dir,
                               f"layout{layout_id}_style{style_id}_seed{seed}")

        if check_collisions:
            scene_model_regex = os.path.join(
                scene_model_root_dir,
                camel_to_snake_case(env_name),
                f"layout{layout_id}_style{style_id}",
                "model/splatfacto/*/nerfstudio_models/",
                f"step-{scene_model_ckpt_idx:09d}.ckpt",
            )
            print(f"Loading scene model from path {scene_model_regex}")
            scene_model_path = list(sorted(glob(scene_model_regex)))[-1]
            scene_model_dir = os.path.dirname(os.path.dirname(scene_model_path))

        os.makedirs(log_dir, exist_ok=True)
        success_history = []

        # populate args and create env
        env_meta_list, shape_meta_list = get_metadata(config)
        env_meta, shape_meta = env_meta_list[0], shape_meta_list[0]
        camera_names = env_meta["env_kwargs"]["camera_names"]
        camera_names += ["robot0_fisheye"]
        override_args = {
            "layout_and_style_ids": [[layout_id, style_id]],
            "camera_names": camera_names,
        }
        if randomize_base_init_pose is not None:
            assert baseline_name == "vanilla_policy"
            override_args["randomize_base_init_pose"] = randomize_base_init_pose
        if not baseline_name == "vanilla_policy":
            override_args["place_robot_for_nav"] = True

        if check_collisions:
            (
                base_fixture_bounds_2d,
                floor_fixture_bounds_2d,
                default_robot_pos,
                default_robot_ori,
                robot_size,
            ) = get_env_map_and_default_robot_init_pose(
                config, override_args, save_dir=log_dir
            )
            ply_path = os.path.join(scene_model_dir, "../../../pc.ply")
            collision_checker = PCVacancyDistribution(
                ply_path,
                floor_fixture_bounds_2d,
                robot_size,
                max_z=1.0,
                buffer_workspace_boundary=True,
            )
        else:
            collision_checker = None

        for i in range(num_episodes if not baseline_name == "vanilla_policy" else 1):
            episode_info = {}

            env = load_env(config, override_args)
            if not baseline_name == "vanilla_policy":
                env.unwrapped.env.place_robot_for_nav_rng = np.random.default_rng(seed * 10000 + i)

                # reset env
                obs = env.reset()

                # get instruction
                instruction = env.unwrapped.env.get_object_name()

                # run nav for a fixed number of steps
                if baseline_name.startswith("lelan"):
                    nav_info = run_lelan_in_env(env, lelan, obs, instruction,
                                                collision_checker=collision_checker)
                    save_video(nav_info["images"], os.path.join(log_dir, f"ep{i}_nav.mp4"))
                    del nav_info["images"]
                    episode_info.update(nav_info)
                else:
                    assert baseline_name == "il_nav"

            # run policy eval
            video_path = os.path.join(log_dir, f"{env_name}_epoch_ep{i}_rollout.mp4")
            all_rollout_logs, _ = TrainUtils.rollout_with_stats(
                policy=rollout_model,
                envs=[env],
                horizon=[config.experiment.rollout.horizon if horizon is None else horizon],
                use_goals=config.use_goals,
                num_episodes=1 if not baseline_name == "vanilla_policy" else num_episodes,
                render=False,
                video_dir=log_dir,
                epoch=f"ep{i}_rollout",
                video_skip=config.experiment.get("video_skip", 5),
                terminate_on_success=config.experiment.rollout.terminate_on_success,
                reset_before_rollout=(baseline_name == "vanilla_policy"),
                del_envs_after_rollouts=True,
                data_logger=None,
            )
            success = all_rollout_logs[env.name]["Success_Rate"]
            if success:
                os.system(f"mv {video_path} {video_path.replace('.mp4', '_s.mp4')}")
            else:
                os.system(f"mv {video_path} {video_path.replace('.mp4', '_f.mp4')}")
            success_history.append(success)
            episode_info.update({"success": success})
            if not baseline_name == "vanilla_policy":
                print(f"Episode {i} ended with [{'success' if success else 'failure'}]. Success so far: {np.sum(success_history)} / {len(success_history)}.")
                np.savez(os.path.join(log_dir, f"ep{i}_info.npz"), **episode_info)
            else:
                print(f"Episodes end with success rate {success}")
                # repeat ep info as we evaluated num_episodes times
                for j in range(num_episodes):
                    np.savez(os.path.join(log_dir, f"ep{j}_info.npz"), **episode_info)


if __name__ == "__main__":
    main()
