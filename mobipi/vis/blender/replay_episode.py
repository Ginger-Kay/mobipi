"""
Replays an episode executed by Mobi-π.

Usage:
    python visualize_mobipi.py \
        --env_name [env_name (e.g. TurnOnStove)] \
        --seed [seed (e.g. 1)] \
        --layout_id [layout ID (e.g. 1)] \
        --style_id [style ID (e.g. 1)] \
        --episode [episode index (e.g. 0)] \
        --replay_root_dir [your custom data output directory]
"""
import os
import sys
import json
import click
import shutil
import numpy as np
import torch
from datetime import datetime
from glob import glob
from mobipi.utils.io_utils import DualStream, camel_to_snake_case
from mobipi.utils.env_utils import load_env, get_env_map_and_default_robot_init_pose
from mobipi.utils.nav_utils import move_to_pose
from mobipi.utils.media_utils import save_video
from mobipi.utils.score_utils import PCVacancyDistribution
from mobipi.utils.policy_utils import get_config_for_policy, load_policy
from mujoco.usd import exporter
from mobipi.macros import POLICY_CKPT_ROOT_DIR, DATA_ROOT_DIR, LOG_ROOT_DIR, SCENE_MODEL_ROOT_DIR


@click.command()
@click.option("--env_name", default="TurnOnSinkFaucet", type=str)
@click.option("--policy_name", default="bc_xfmr", type=str)
@click.option("--seed", default=1, type=int)
@click.option("--layout_id", default=1, type=int)
@click.option("--style_id", default=1, type=int)
@click.option("--scene_id", default=0, type=int)
@click.option("--episode", "-e", required=True, type=int,
              help="Episode index to replay (matches ep{episode}_info.npz)")
@click.option("--log_root_dir", default=LOG_ROOT_DIR, type=str,
              help="Base directory where original logs are stored")
@click.option("--scene_model_root_dir", default=SCENE_MODEL_ROOT_DIR, type=str)
@click.option("--scene_model_ckpt_idx", default=29999, type=int)
@click.option("--replay_root_dir", default=None, type=str,
              help="Optional parent directory to store replay outputs."
                   " If not set, defaults to same parent as original logs.")
def main(env_name, policy_name, seed, layout_id, style_id, scene_id,
         episode, log_root_dir, scene_model_root_dir,
         scene_model_ckpt_idx, replay_root_dir):
    # Determine parent directories
    method_name = "mobpi_v12.1_rrt" # "mobipi"
    log_parent = os.path.join(log_root_dir, env_name, method_name, policy_name)
    default_parent = os.path.join(
        log_parent,
        f"layout{layout_id}_style{style_id}_seed{seed}"
    )
    parent_dir = replay_root_dir or default_parent

    # Compose replay subdirectory name including env, layout, seed, episode
    subdir_name = f"replay_{env_name}_layout{layout_id}_style{style_id}_seed{seed}_ep{episode}"
    replay_dir = os.path.join(parent_dir, subdir_name)
    usd_dir = os.path.join(replay_dir, "usd")

    # Create directories
    os.makedirs(os.path.join(parent_dir, "success"), exist_ok=True)
    os.makedirs(os.path.join(parent_dir, "fail"), exist_ok=True)
    os.makedirs(usd_dir, exist_ok=True)
    os.makedirs(replay_dir, exist_ok=True)

    # Load episode info
    info_path = os.path.join(default_parent, f"ep{episode}_info.npz")
    data = np.load(info_path, allow_pickle=True)
    if data["success"].item() < 1:
        print(f"Exiting env {env_name} layout {layout_id} style {style_id} because episode is not successful.")
        shutil.rmtree(replay_dir)
        exit(0)

    # Redirect logs
    log_file = os.path.join(replay_dir, f"replay_{episode}.log")
    sys.stdout = DualStream(open(log_file, "a"), sys.stdout)
    sys.stderr = DualStream(open(log_file, "a"), sys.stderr)

    sampled_points = data["sampled_points"]
    sampled_scores = data["sampled_scores"]
    best_idx = np.argmax(sampled_scores)
    best_pose = sampled_points[best_idx]
    print(f"Selected best pose: {best_pose}")

    # Load policy
    config, ckpt_path = get_config_for_policy(
        POLICY_CKPT_ROOT_DIR, DATA_ROOT_DIR,
        env_name, policy_name, seed=seed,
        dataset_name="mg-300"
    )
    model, rollout_model, _ = load_policy(config, ckpt_path)

    # Find scene model dir
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

    # Create env
    override_args = {
        "layout_and_style_ids": [[layout_id, style_id]],
        "place_robot_for_nav": True,
        "hard_reset": True,
        "force_robot_placement": (np.array([data["base_vec_history"][0][0], data["base_vec_history"][0][1], 0]),
                                  np.array([0, 0, data["base_vec_history"][0][-1]]))
    }
    env = load_env(config, override_args)

    (
        base_fixture_bounds_2d,
        floor_fixture_bounds_2d,
        default_robot_pos,
        default_robot_ori,
        robot_size,
    ) = get_env_map_and_default_robot_init_pose(
        config, override_args
    )

    # Initialize collision avoidance function
    ply_path = os.path.join(os.path.join(scene_model_dir, "../../../pc.ply"))
    k_col = PCVacancyDistribution(ply_path, floor_fixture_bounds_2d, robot_size)

    # Initial reset
    obs = env.reset()

    # Start USD recording
    width, height, max_geom = 640, 480, 1000
    exp = exporter.USDExporter(
        env.unwrapped.env.sim.model._model,
        max_geom=500_000,
        output_directory_root=replay_dir,
        output_directory="usd",
        light_intensity=1000.,
        camera_names=["robot0_agentview_left", "robot0_agentview_center",
                      "robot0_agentview_right"]
    )

    def step_callback():
        exp.update_scene(env.unwrapped.env.sim.data._data)
        print(f"sim step {env.timestep}")

    # Navigate to best base pose
    from scipy.spatial.transform import Rotation as R
    trans = np.eye(4)
    trans[0:3, 3] = [best_pose[0], best_pose[1], 0]
    yaw = best_pose[2]
    trans[0:3, 0:3] = R.from_euler('z', yaw).as_matrix()
    nav_info = move_to_pose(env, trans, render=True,
                            verbose=True, use_rrt=True,
                            k_col=k_col, step_callback=step_callback,
                            legacy=False)
    save_video(nav_info["images"], os.path.join(replay_dir, f"episode{episode}_nav.mp4"))
    del nav_info["images"]
    np.savez(os.path.join(replay_dir, f"ep{episode}_info.npz"), **nav_info)

    # Manual policy rollout loop
    horizon = config.experiment.rollout.horizon
    frames = []
    obs = env._get_stacked_obs_from_history()
    done = False
    rollout_model.start_episode(lang=env._ep_lang_str)
    goal = None
    if config.use_goals:
        goal = env.get_goal()
    for t in range(horizon):
        with torch.no_grad():
            action = rollout_model(ob=obs, goal=goal)
        obs, reward, done, info = env.step(action)
        img = obs["robot0_agentview_left_image"][-1].transpose(1, 2, 0)
        frames.append((img * 255).astype(np.uint8))
        step_callback()
        if done:
            break

    # Save manipulation video
    manip_vid = os.path.join(replay_dir, f"episode{episode}_manip.mp4")
    save_video(frames, manip_vid)

    # Save scene
    exp.save_scene(filetype="usd")

    is_success = info["is_success"]["task"]
    if is_success:
        rename_dir = os.path.join(parent_dir, "success", subdir_name)
        shutil.move(replay_dir, rename_dir)
    else:
        rename_dir = os.path.join(parent_dir, "fail", subdir_name)
        shutil.move(replay_dir, rename_dir)

    print("Replay complete")


if __name__ == "__main__":
    main()
