"""
After obtaining the manipulation policy and 3DGS models of target envs, run
this script to execute and evaluate our method.

Usage:
python eval_mobipi.py --env_name CloseSingleDoor --scene_ids -1 --seed 1 --vis

@yjy0625
"""
import os
import sys
import json
import h5py
import copy
import torch
import click
import warnings
import numpy as np
from datetime import datetime
from glob import glob
from tqdm import tqdm
from torchvision import transforms
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
import robosuite.utils.transform_utils as T
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix
import robomimic.utils.train_utils as TrainUtils
import mimicgen.utils.pose_utils as PoseUtils

from mobipi.macros import *
from mobipi.utils.constants import *
from mobipi.utils.io_utils import DualStream, camel_to_snake_case
from mobipi.utils.env_utils import (
    get_metadata,
    load_env,
    get_env_map_and_default_robot_init_pose,
    adjust_initial_robot_position,
    sample_robot_positions,
    sample_robot_orientations,
)
from mobipi.utils.media_utils import save_video
from mobipi.utils.policy_utils import get_config_for_policy, load_policy
from mobipi.utils.handoff_state import (
    capture_handoff_snapshot,
    write_handoff_snapshot,
)
from mobipi.utils.controller_state import env_controller_adapters
from mobipi.scene_model.scene_model import BatchSceneModel
from mobipi.utils.encoder_utils import DinoDenseDescriptorEncoder, DinoEncoder, PolicyEncoder
from mobipi.utils.score_utils import HybridDistribution
from mobipi.utils.opt_utils import optimize_pose_batch

from mobipi.utils.opt_utils import normalize as normalize_in_bounds
from mobipi.utils.nav_utils import move_to_pose
from mobipi.utils.env_utils import check_robot_positions
from mobipi.utils.env_utils import (
    compute_robot_angle_from_mat,
    compute_relative_cam_pose,
    compute_camera_extrinsics,
    render_image_with_robot_mask,
)
from mobipi.utils.vis_utils import (
    plot_best_metrics,
    get_topdown_image_with_corners,
    plot_scores_on_topdown_image,
    plot_pose_histograms,
    plot_best_renders,
)


import warnings
warnings.filterwarnings(
    "ignore",
    message=".*default value of the antialias parameter.*",
    category=UserWarning,
)


@click.command()
@click.option("--env_name", default="TurnOnSinkFaucet", type=str)
@click.option("--policy_name", default="bc_xfmr", type=str)
@click.option("--seed", default=1, type=int)
@click.option("--scene_ids", default=None, type=str)
@click.option("--layout_id", default=1, type=int)
@click.option("--style_id", default=1, type=int)
@click.option("--env_seeds", default=None, type=str)
@click.option("--ckpt_root_dir", default=POLICY_CKPT_ROOT_DIR, type=str)
@click.option("--data_root_dir", default=DATA_ROOT_DIR, type=str)
@click.option("--log_root_dir", default=LOG_ROOT_DIR, type=str)
@click.option("--scene_model_root_dir", default=SCENE_MODEL_ROOT_DIR, type=str)
@click.option("--scene_model_ckpt_idx", default=29999, type=int)
@click.option("--num_episodes", default=10, type=int)
@click.option("--bo_num_samples", default=500, type=int)
@click.option("--debug", is_flag=True)
@click.option("--vis", is_flag=True)
@click.option("--k", default=5, type=int)
@click.option("--skip_detection", is_flag=True)
@click.option("--skip_collision", is_flag=True)
@click.option("--encoder_name", default="dino_dense_descriptor")
@click.option("--detector_name", default="openbmb/MiniCPM-V-2")
@click.option("--detector_view_robot", is_flag=True)
@click.option("--base_estimator", default="ET")
@click.option("--num_init_samples", default=2500)
@click.option(
    "--ec1-handoff-dir",
    default=None,
    type=str,
    help=(
        "Write a scoped post-navigation EC-1 handoff snapshot here; "
        "this schema does not claim exhaustive state coverage. "
        "The option is fail-closed and requires explicit controller state support."
    ),
)
def main(
    env_name,
    policy_name,
    seed,
    scene_ids,
    layout_id,
    style_id,
    env_seeds,
    ckpt_root_dir,
    data_root_dir,
    log_root_dir,
    scene_model_root_dir,
    scene_model_ckpt_idx,
    num_episodes,
    bo_num_samples,
    debug,
    vis,
    k,
    skip_detection,
    skip_collision,
    encoder_name,
    detector_name,
    detector_view_robot,
    base_estimator,
    num_init_samples,
    ec1_handoff_dir,
):
    # Configure run name
    method_name = "mobipi_debug" if debug else f"mobipi"
    if detector_name != "openbmb/MiniCPM-V-2":
        method_name += f"det-{detector_name}"
    if skip_collision:
        method_name += "_wo-col"
    if skip_detection:
        method_name += "_wo-det"
    if encoder_name == "dino_dense_descriptor":
        pass
    elif encoder_name == "policy":
        method_name += "_env-policy"
    elif encoder_name == "dino":
        method_name += "_enc-dino1d"
    else:
        raise ValueError(f"Unrecognized encoder name [{encoder_name}]")
    if base_estimator != "ET":
        method_name += "_est-{base_estimator}"
    if detector_view_robot:
        method_name += "_detviewrobot"
    if bo_num_samples == 0:
        method_name += "_wo-bo"
    if num_init_samples != 2500:
        method_name += f"_ninit-{num_init_samples}"
    log_parent_dir = os.path.join(log_root_dir, env_name, method_name, policy_name)

    # Create a timestamped log file
    os.makedirs(log_parent_dir, exist_ok=True)
    log_file = os.path.join(
        log_parent_dir, f"console_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_stream = open(log_file, "a")

    # Replace stdout and stderr
    sys.stdout = DualStream(file_stream, sys.stdout)
    sys.stderr = DualStream(file_stream, sys.stderr)

    # Load policy
    config, ckpt_path = get_config_for_policy(
        ckpt_root_dir, data_root_dir, env_name, policy_name, seed=seed,
        dataset_name="mg-300"
    )
    model, rollout_model, lang_encoder = load_policy(config, ckpt_path)

    # Load encoder used in score function
    if encoder_name == "dino_dense_descriptor":
        encoder = DinoDenseDescriptorEncoder(
            device="cuda", model_name="vit_base_patch16_224_dino"
        )
    elif encoder_name == "policy":
        encoder = PolicyEncoder(config, ckpt_path, device="cuda")
    else:
        encoder = DinoEncoder(device="cuda", model_name="vit_base_patch16_224_dino")

    if scene_ids is not None:
        scene_ids_nums = [int(s.strip()) for s in scene_ids.split(",")]
        layout_and_style_ids = [[i, i] for i in scene_ids_nums]
    elif layout_id >= 0:
        layout_and_style_ids = [[layout_id, style_id]]
    else:
        layout_and_style_ids = [[i, i] for i in TEST_LAYOUTS]

    all_default_poses, all_selected_poses = [], []
    for layout_style_env_id in layout_and_style_ids:
        layout_id, style_id = layout_style_env_id
        print(f"--- Evaluating layout {layout_id} style {style_id} ---")

        # find scene model ckpt dir
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

        # Create log directory
        log_dir = os.path.join(
            log_parent_dir,
            f"layout{layout_id}_style{style_id}_seed{seed}"
        )
        os.makedirs(log_dir, exist_ok=True)
        success_history = []

        # Populate args and create env
        print(f"Preparing env configs.")
        env_meta_list, shape_meta_list = get_metadata(config)
        env_meta, shape_meta = env_meta_list[0], shape_meta_list[0]
        override_args = {
            "layout_and_style_ids": [[layout_id, style_id]],
            "place_robot_for_nav": True,
            "camera_names": env_meta["env_kwargs"]["camera_names"] + ["robot0_navview"],
            "hard_reset": True
        }

        (
            base_fixture_bounds_2d,
            floor_fixture_bounds_2d,
            default_robot_pos,
            default_robot_ori,
            robot_size,
        ) = get_env_map_and_default_robot_init_pose(
            config, override_args, save_dir=log_dir
        )

        # Initialize object for computing scores
        print(f"Loading score function.")
        dist = HybridDistribution(
            config,
            encoder,
            k=k,
            object_detection_model_name=detector_name,
            ply_path=os.path.join(scene_model_dir, "../../../pc.ply"),
            floor_fixture_bounds_2d=floor_fixture_bounds_2d,
            robot_size=robot_size,
            max_z=1.0,
            image_size=224,
            skip_detection=skip_detection,
            skip_collision=skip_collision,
            detector_view_robot=detector_view_robot
        )

        for i in range(num_episodes):
            episode_info = {}
            rng = np.random.RandomState(seed * 10000 + i)
            env = load_env(config, override_args)
            env.unwrapped.env.place_robot_for_nav_rng = np.random.default_rng(
                seed * 10000 + i
            )
            print("Initialized env. Performing reset.")

            # reset env
            obs = env.reset()
            print("Reset done.")

            sampled_init_pos = env.unwrapped.env._init_robot_pos
            sampled_init_ori = env.unwrapped.env._init_robot_ori

            if env_name == "CloseDrawer":
                camera_names = ["robot0_agentview_right", "robot0_agentview_left"]
            else:
                camera_names = ["robot0_agentview_right"]
            obj_name = env.unwrapped.env.get_object_name()
            dist.set_obs_keys([camera_name + "_image" for camera_name in camera_names])
            dist.set_object(obj_name)
            dist.set_task_name(env.unwrapped.env.get_ep_meta()["lang"])
            print(f"Setting cameras to {camera_names} and obj name to [{obj_name}]")

            image_width, image_height = 224, 224
            camera_intrinsics_list = [
                get_camera_intrinsic_matrix(
                    env.unwrapped.env.sim,
                    camera_name,
                    camera_height=image_height,
                    camera_width=image_width,
                )
                for camera_name in camera_names
            ]
            camera_intrinsics_dict = [
                {
                    "w": image_width,
                    "h": image_height,
                    "fl_x": camera_intrinsics[0, 0],
                    "fl_y": camera_intrinsics[1, 1],
                    "cx": camera_intrinsics[0, -1],
                    "cy": camera_intrinsics[1, -1],
                }
                for camera_intrinsics in camera_intrinsics_list
            ]

            scene_model = BatchSceneModel(scene_model_dir, camera_intrinsics_dict)

            base_pos = env.unwrapped.env.sim.data.get_body_xpos("mobilebase0_base")
            base_heading = compute_robot_angle_from_mat(
                env.unwrapped.env.sim.data.get_body_xmat("mobilebase0_base").reshape(
                    3, 3
                )
            )

            # Define score function wrapper used in optimization algo
            def score_function(
                robot_poses,
                robot_imgs_torch,
                robot_masks_torch,
                rel_cam_positions,
                rel_cam_mats,
                base_pos,
                base_heading,
            ):
                num_poses = len(robot_poses)

                def render_images(robot_pose):
                    """Render the images for `k_id`."""
                    with torch.no_grad():
                        robot_pose_torch = torch.tensor(
                            robot_pose, device="cuda", dtype=torch.float32
                        )
                        batch_view_extrinsics = [
                            compute_camera_extrinsics(
                                robot_pose_torch, rel_cam_position, rel_cam_mat
                            )
                            for rel_cam_position, rel_cam_mat in zip(
                                rel_cam_positions, rel_cam_mats
                            )
                        ]
                        batch_rendered_images = scene_model.render(
                            batch_view_extrinsics, image_size=224
                        )
                        batch_edited_rendered_images = (
                            batch_rendered_images * (1 - robot_masks_torch)
                            + robot_imgs_torch * robot_masks_torch
                        )
                    return batch_rendered_images, batch_edited_rendered_images

                # Create lazy rendering functions for each robot pose
                lazy_renders = [
                    lambda pose=pose: render_images(pose) for pose in robot_poses
                ]

                scores, scores_per_view, padded_rendered_images = dist.compute_score(
                    lazy_renders,
                    torch.tensor(robot_poses, device="cuda", dtype=torch.float32),
                    env=env,
                    base_pos=base_pos,
                    base_heading=base_heading,
                    check_fov=np.pi * 75 / 180,
                )

                score_info = {
                    "scores_per_view": [
                        s.detach().cpu().numpy().tolist() for s in scores_per_view
                    ],
                    "rendered_images": [im for im in padded_rendered_images],
                }

                return scores.detach().cpu().numpy().tolist(), score_info

            # Setup optimization bounds
            bounds = [
                (floor_fixture_bounds_2d[2, 0], floor_fixture_bounds_2d[0, 0]),
                (floor_fixture_bounds_2d[0, 1], floor_fixture_bounds_2d[1, 1]),
                (base_heading - np.pi / 2, base_heading + np.pi / 2),
            ]

            # Retrieve camera information
            rel_cam_positions, rel_cam_mats, robot_imgs, robot_masks = [], [], [], []
            for camera_name in camera_names:
                (rel_cam_position, rel_cam_mat, robot_img, robot_mask) = (
                    render_image_with_robot_mask(
                        env, image_height, image_width, camera_name=camera_name
                    )
                )
                rel_cam_positions.append(rel_cam_position)
                rel_cam_mats.append(rel_cam_mat)
                robot_imgs.append(robot_img)
                robot_masks.append(robot_mask)
            robot_imgs_torch = torch.tensor(
                np.array(robot_imgs) / 255,
                dtype=torch.float32,
                requires_grad=True,
                device="cuda",
            )
            robot_masks_torch = torch.tensor(
                np.array(robot_masks)[..., None],
                dtype=torch.float32,
                requires_grad=True,
                device="cuda",
            )
            rel_cam_positions = torch.tensor(
                np.array(rel_cam_positions), dtype=torch.float32, device="cuda"
            )
            rel_cam_mats = torch.tensor(
                np.array(rel_cam_mats), dtype=torch.float32, device="cuda"
            )

            warnings.filterwarnings(
                "ignore",
                message="The default value of the antialias parameter of all the resizing transforms",
            )

            # Calculate distance from current robot pose to the boundary
            # Then use this for sampling
            dist_to_xmin = base_pos[0] - bounds[0][0]
            dist_to_xmax = bounds[0][1] - base_pos[0]
            dist_to_ymin = base_pos[1] - bounds[1][0]
            dist_to_ymax = bounds[1][1] - base_pos[1]
            dist_to_boundary = np.minimum(
                np.where(
                    np.cos(base_heading) > 0,
                    dist_to_xmax / np.cos(base_heading),
                    dist_to_xmin / -np.cos(base_heading),
                ),
                np.where(
                    np.sin(base_heading) > 0,
                    dist_to_ymax / np.sin(base_heading),
                    dist_to_ymin / -np.sin(base_heading),
                ),
            )

            normalized_initial_samples = rng.uniform(0, 1, size=(num_init_samples, 3))

            val2str = lambda a: f"[{a[0]:.2f}, {a[1]:.2f}, {a[2]:.2f}]"

            # Run optimization
            best_pose, history = optimize_pose_batch(
                lambda p: score_function(
                    p,
                    robot_imgs_torch,
                    robot_masks_torch,
                    rel_cam_positions,
                    rel_cam_mats,
                    base_pos,
                    base_heading,
                ),
                bounds,
                algorithm="bayesian",
                base_estimator=base_estimator,
                n_iterations=bo_num_samples // 5,
                normalize_data=True,
                track_info=True,
                initial_samples=normalized_initial_samples,
                acq_func="LCB",
                acq_func_kwargs=dict(kappa=1.96),
                seed=seed * 10000,
                batch_size=5,
                print_fn=val2str,
            )

            episode_info.update(history)
            print(f"Best pose: {best_pose.round(2)}")

            default_pose = np.concatenate(
                [default_robot_pos[:2], [default_robot_ori[-1]]]
            )
            all_default_poses.append(default_pose)
            all_selected_poses.append(best_pose)

            # Visualizations
            sampled_points = history["sampled_points"]

            if len(history["rendered_images"]) > 0:
                plot_best_renders(
                    np.array(history["rendered_images"]),  # (N, num_views, H, W, 3)
                    history["sampled_scores"],  # (N,)
                    save_path=os.path.join(log_dir, f"ep{i}_renders.pdf"),
                )
                del history["rendered_images"]

            plot_best_metrics(
                default_pose,
                sampled_points,
                np.array(history["sampled_scores"]),
                save_path=os.path.join(log_dir, f"ep{i}_progress.pdf"),
            )
            best_pose_mat = PoseUtils.make_pose(
                np.concatenate([best_pose[:2], [0]]),
                T.quat2mat(T.axisangle2quat([0, 0, best_pose[2]])),
            )

            # Navigate to predicted best base pose
            nav_info = move_to_pose(
                env,
                best_pose_mat,
                render=True,
                verbose=False,
                use_rrt=True,
                k_col=dist.k_col,
            )
            if ec1_handoff_dir is not None:
                controller_get_state, controller_set_state = env_controller_adapters(env)
                handoff_path = os.path.join(
                    ec1_handoff_dir,
                    env_name,
                    f"layout{layout_id}_style{style_id}_seed{seed}_ep{i}.pkl",
                )
                handoff_snapshot = capture_handoff_snapshot(
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
                        "evaluator": "mobipi/eval/eval_mobipi.py",
                        "handoff": "post_navigation_pre_policy_rollout",
                        "env_name": env_name,
                        "layout_id": layout_id,
                        "style_id": style_id,
                        "seed": seed,
                        "episode_index": i,
                    },
                )
                write_handoff_snapshot(handoff_path, handoff_snapshot)
                print(f"Wrote EC-1 handoff snapshot to {handoff_path}")
            save_video(nav_info["images"], os.path.join(log_dir, f"ep{i}_nav.mp4"))

            del nav_info["images"]
            episode_info.update(nav_info)

            # Run policy rollout after navigation
            video_path = os.path.join(log_dir, f"{env_name}_epoch_ep{i}_rollout.mp4")
            all_rollout_logs, _ = TrainUtils.rollout_with_stats(
                policy=rollout_model,
                envs=[env],
                horizon=[config.experiment.rollout.horizon],
                use_goals=config.use_goals,
                num_episodes=1,
                render=False,
                video_dir=log_dir,
                epoch=f"ep{i}_rollout",
                video_skip=config.experiment.get("video_skip", 5),
                terminate_on_success=config.experiment.rollout.terminate_on_success,
                reset_before_rollout=False,
                del_envs_after_rollouts=True,
                data_logger=None,
            )

            # Bookkeeping
            success = all_rollout_logs[env.name]["Success_Rate"]
            success_history.append(success)
            episode_info.update({"success": success})
            if "rendered_images" in episode_info:
                del episode_info["rendered_images"]
            print(
                f"Episode {i} ended with [{'success' if success else 'failure'}]. Success so far: {np.sum(success_history)} / {len(success_history)}."
            )
            if success:
                os.system(f"mv {video_path} {video_path.replace('.mp4', '_s.mp4')}")
            else:
                os.system(f"mv {video_path} {video_path.replace('.mp4', '_f.mp4')}")
            open(os.path.join(log_dir, f"ep{i}_{'s' if success else 'f'}.txt"), 'w').close()
            np.savez(os.path.join(log_dir, f"ep{i}_info.npz"), **episode_info)

            # Visualizations
            if vis:
                topdown_image, image_corners = get_topdown_image_with_corners(
                    config, override_args, bounds
                )

                plot_scores_on_topdown_image(
                    topdown_image,
                    image_corners,
                    bounds,
                    sampled_points,
                    np.array(history["sampled_scores"]),
                    vis_size=0.05,
                    save_path=os.path.join(log_dir, f"ep{i}_topdown_scores_{'s' if success else 'f'}.pdf"),
                    render_heading=True,
                    default_pose=default_pose,
                    nav_pose=nav_info["base_vec_history"][-1],
                    init_pose=np.array(
                        [sampled_init_pos[0], sampled_init_pos[1], sampled_init_ori[-1]]
                    ),
                    selected_pose=best_pose,
                )

            del scene_model

        del dist

        # Plot stats
        plot_pose_histograms(
            np.array(all_default_poses),
            np.array(all_selected_poses),
            os.path.join(log_dir, "hist.pdf"),
        )


if __name__ == "__main__":
    main()
