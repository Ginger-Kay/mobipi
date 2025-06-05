"""
Plots achieved navigation end poses of competing methods.

Usage:
python plot_nav_poses.py

@yjy0625
"""
import os
import re

import json
import torch
import click
import numpy as np
from glob import glob
from tqdm import tqdm
from torchvision import transforms
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from scipy.spatial.transform import Rotation as R
from robosuite.utils.transform_utils import mat2quat
from robosuite.utils.mjmod import CameraModder
from robocasa.utils.env_utils import create_env

from mobipi.macros import *
from mobipi.utils.vis_utils import plot_pose
from mobipi.utils.env_utils import load_env, get_env_map_and_default_robot_init_pose


def setup_orthographic_camera(env, camera_name, xy_center, z_height, rotation_angle, fovy=0.2):
    """
    Set up an orthographic top-down camera in Robosuite.

    Args:
        env (MujocoEnv): Robosuite environment.
        camera_name (str): Name of the camera.
        xy_center (tuple): (x, y) center of the scene in world coordinates.
        z_height (float): Height of the camera in the z-axis.
        rotation_angle (float): Rotation around z-axis in radians.

    Returns:
        dict: Camera parameters including intrinsics and extrinsics.
    """
    sim = env.sim
    camera_modder = CameraModder(sim)
    camera_id = sim.model.camera_name2id(camera_name)

    # Set camera position
    camera_pos = np.array([xy_center[0], xy_center[1], z_height])
    camera_modder.set_pos(camera_name, camera_pos)

    # Set camera rotation (rotation about z-axis)
    rot_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
        [0, np.sin(rotation_angle), np.cos(rotation_angle)],
    ])
    camera_quat = mat2quat(rot_matrix)
    camera_modder.set_quat(camera_name, camera_quat)

    # Set orthographic projection
    sim.model.cam_fovy[camera_id] = fovy

    sim.forward()

    # Get camera parameters
    camera_params = {
        "pos": camera_pos,
        "quat": camera_quat,
        "image_size": image_size,
        "fovy": fovy,
    }
    return camera_params


def unproject_image_corners_on_floor(cam_pos, cam_quat, cam_fovy, image_size):
    """
    Unprojects the image corners onto the z=0 plane in world coordinates.

    Args:
        cam_pos (np.array): Camera position (x, y, z) as a numpy array of shape (3,).
        cam_quat (np.array): Camera orientation quaternion (x, y, z, w) as a numpy array of shape (4,).
        cam_fovy (float): Field of view in the y direction, in degrees.
        image_size (tuple): Image size as (width, height).

    Returns:
        np.array: Unprojected (x, y) world coordinates of the four image corners.
    """
    # Unpack image size
    img_width, img_height = image_size

    # Calculate aspect ratio and field of view in radians
    aspect_ratio = img_width / img_height
    fovy_rad = np.radians(cam_fovy)
    fovx_rad = 2 * np.arctan(np.tan(fovy_rad / 2) * aspect_ratio)

    # Define normalized device coordinates for image corners
    # In NDC: (x, y) ranges from -1 to 1
    ndc_corners = np.array([
        [-1, -1],  # Bottom-left
        [1, -1],   # Bottom-right
        [1, 1],    # Top-right
        [-1, 1]    # Top-left
    ])

    # Calculate the camera's focal lengths in terms of z=1
    focal_length_y = 1 / np.tan(fovy_rad / 2)
    focal_length_x = 1 / np.tan(fovx_rad / 2)

    # Map NDC to camera space at z=-1
    camera_space_corners = np.array([
        [corner[0] / focal_length_x, corner[1] / focal_length_y, -1]
        for corner in ndc_corners
    ])

    # Rotate corners into world space using camera orientation
    rotation_matrix = R.from_quat(cam_quat[[3, 0, 1, 2]]).as_matrix()
    world_space_corners = (rotation_matrix @ camera_space_corners.T).T

    # Calculate intersection with z=0 plane for each corner
    world_xy_corners = []
    for corner in world_space_corners:
        # Ray origin is the camera position
        ray_origin = cam_pos
        # Ray direction is the corner vector from the camera position
        ray_dir = corner / np.linalg.norm(corner)

        # Find intersection with the z=0 plane
        t = -ray_origin[2] / ray_dir[2]  # z=0 plane intersection
        intersection = ray_origin + t * ray_dir

        # Store the (x, y) coordinates of the intersection
        world_xy_corners.append(intersection[:2])

    return np.array(world_xy_corners)


env_name = "TurnOnMicrowave"
methods = ["lelan", "vlfm", "mobipi"]
colors = ["#1abc9c", "#3498db", "#ffa556"]
display_names = ["LeLaN", "VLFM", "Ours"]
policy_name = "bc_xfmr"
layout_id = 1
seeds = [1, 2, 3]
camera_name = "robot0_robotview"
xy_center = (0.9, -0.9)
z_height = 4
rotation_angle = 0
image_size = (2048, 1024)


# plot a top-down map of the scene
env = create_env(env_name=env_name, seed=0, layout_ids=layout_id, style_ids=layout_id, place_robot=False)
base_fixture_bounds, floor_fixture_bounds = env._get_env_map()

camera_params = setup_orthographic_camera(env, camera_name, xy_center, z_height, rotation_angle / 180 * np.pi, fovy=12.)
image = env.sim.render(width=image_size[0], height=image_size[1], camera_name=camera_name)[::-1]

image_corners = unproject_image_corners_on_floor(camera_params["pos"], camera_params["quat"], camera_params["fovy"], image_size)

final_base_pose_data = dict()
for method, color in zip(methods, colors):
    final_base_pose_data[method] = dict(color=color, poses=[], successes=[])
    for seed in seeds:
        npz_dir = os.path.join(LOG_ROOT_DIR, env_name, method, policy_name, f"layout{layout_id}_style{layout_id}_seed{seed}")
        for ep_idx in range(10):
            ep_data = np.load(os.path.join(npz_dir, f"ep{ep_idx}_info.npz"))
            if method.startswith("lelan"):
                assert "base_pos_history" in ep_data
                nav_pos = ep_data["base_pos_history"][-1]
                nav_mat = ep_data["base_mat_history"][-1]
                nav_rot = R.from_matrix(nav_mat).as_rotvec()[-1]
                nav_pose = np.array([nav_pos[0], nav_pos[1], nav_rot])
            elif method == "mobipi":
                assert "base_vec_history" in ep_data
                nav_pose = ep_data["base_vec_history"][-1]
            elif method == "vlfm_debug":
                assert "base_pos_history" in ep_data
                nav_pos = ep_data["base_pos_history"][-1]
                nav_mat = ep_data["base_mat_history"][-1]
                nav_rot = R.from_matrix(nav_mat).as_rotvec()[-1]
                # print(ep_data["base_pos_history"])
                nav_pose = np.array([nav_pos[0], nav_pos[1], nav_rot])
            else:
                raise NotImplementedError()
            final_base_pose_data[method]["poses"].append(nav_pose)
            final_base_pose_data[method]["successes"].append(ep_data["success"])

fig, ax = plt.subplots(figsize=(8, 4))
ax.imshow(image, extent=(image_corners[0, 0], image_corners[1, 0], image_corners[2, 1], image_corners[0, 1]),
          zorder=1)
ax.set_aspect("equal")
success_idx = 0
for method, data in final_base_pose_data.items():
    for i, pose in enumerate(data["poses"]):
        success = data["successes"][i]
        plot_pose(ax, pose, vis_size=0.08, color=data["color"], alpha=1.0 if success else 0.3,
                  edgecolor="black", linewidth=2, zorder=1 if not success else 2 + success_idx)
        if success:
            success_idx += 1

legend_elements = [
    Patch(facecolor=color, edgecolor='black', label=name)
    for color, name in zip(colors, display_names)
]
ax.legend(handles=legend_elements, loc="lower center", ncol=3,
          fontsize=14,           # larger text
          handleheight=1.,       # taller legend handles
          handlelength=1.,       # wider legend handles
          borderpad=0.6,         # more space around legend content
          labelspacing=.4        # more vertical space between labels
         )
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])

plt.tight_layout()
fig.savefig("sim_qual.pdf", bbox_inches='tight', pad_inches=0.0)
