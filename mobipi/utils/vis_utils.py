"""
Utilities for visualization.

@yjy0625
"""
import os
import json
import numpy as np
import torch
import pickle
from scipy.spatial.transform import Rotation as R

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
import matplotlib.patches as patches
import matplotlib.cm as cm

from robosuite.utils.transform_utils import mat2quat
from robosuite.utils.mjmod import CameraModder
from robomimic.config import config_factory
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils

from mobipi.utils.env_utils import load_env


def setup_orthographic_camera(env, camera_name, xy_center, z_height, rotation_angle, image_size=2048, fovy=0.2):
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


def get_topdown_image_with_corners(config, override_args, bounds, camera_name="freeview", image_size=2048):
    xy_center = ((bounds[0][0] + bounds[0][1]) / 2, (bounds[1][0] + bounds[1][1]) / 2)  # Center of the scene
    z_height = 450
    rotation_angle = 180

    env = load_env(config, override_args)
    env.reset()
    camera_params = setup_orthographic_camera(
        env.unwrapped.env, camera_name, xy_center, z_height, rotation_angle / 180 * np.pi, fovy=1.0,
        image_size=image_size
    )
    image = env.unwrapped.env.sim.render(width=image_size, height=image_size,
                                         camera_name=camera_name)[::-1]

    # compute 2d locations of image corners
    image_corners = unproject_image_corners_on_floor(camera_params["pos"], camera_params["quat"],
                                                     camera_params["fovy"], (image_size, image_size))
    return image, image_corners


def plot_pose(ax, pose, vis_size, color, alpha, render_heading=True, edgecolor=None, linewidth=0.0,
              hatch=None, zorder=None):
    if render_heading:
        # Tip of the triangle
        tip_x = pose[0] + vis_size * np.cos(pose[2])
        tip_y = pose[1] + vis_size * np.sin(pose[2])

        # Base of the triangle
        base_left_x = pose[0] - vis_size * 0.5 * np.cos(pose[2]) - vis_size * 0.5 * np.sin(pose[2])
        base_left_y = pose[1] - vis_size * 0.5 * np.sin(pose[2]) + vis_size * 0.5 * np.cos(pose[2])

        base_right_x = pose[0] - vis_size * 0.5 * np.cos(pose[2]) + vis_size * 0.5 * np.sin(pose[2])
        base_right_y = pose[1] - vis_size * 0.5 * np.sin(pose[2]) - vis_size * 0.5 * np.cos(pose[2])

        # Create the triangle
        triangle = patches.Polygon(
            [
                (tip_x, tip_y),
                (base_left_x, base_left_y),
                (pose[0], pose[1]),
                (base_right_x, base_right_y)
            ],
            closed=True,
            facecolor=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            hatch=hatch,
            zorder=zorder
        )
        ax.add_patch(triangle)


def plot_scores_on_topdown_image(
    topdown_image,
    image_corners,
    bounds,
    poses,
    scores,
    default_pose=None,
    init_pose=None,
    selected_pose=None,
    nav_pose=None,
    vis_size=0.05,
    save_path=None,
    render_heading=False,
    return_fig=False
):
    if save_path is not None:
        plot_data = dict(
            topdown_image=topdown_image,
            image_corners=image_corners,
            bounds=bounds,
            poses=poses,
            scores=scores,
            default_pose=default_pose,
            init_pose=init_pose,
            selected_pose=selected_pose,
            nav_pose=nav_pose,
            vis_size=vis_size,
            save_path=save_path,
            render_heading=render_heading
        )
        np.savez(save_path.replace(".pdf", "_plot_data.npz"), **plot_data)

    # Define custom colors for -1 and -2 scores
    color_negative_one = "black"
    color_negative_two = "red"
    color_negative_three = "gray"

    if len(scores) > 0:
        # Separate scores into categories
        positive_scores = scores[scores > 0]
        if len(positive_scores) > 0:
            normalized_scores = (positive_scores - positive_scores.min()) / (positive_scores.max() - positive_scores.min())
        else:
            normalized_scores = positive_scores

        # Define colormap for positive scores
        viridis_cmap = cm.viridis
        if len(positive_scores) > 0:
            norm = Normalize(vmin=positive_scores.min(), vmax=positive_scores.max())
        else:
            norm = Normalize(vmin=0, vmax=1)

    # Set up the figure with two subplots
    fig, axes = plt.subplots(1, 1, figsize=(10, 7))
    ax2 = axes

    # Right plot: top-down view with scores
    ax2.imshow(topdown_image, extent=(image_corners[0, 0], image_corners[1, 0], image_corners[2, 1], image_corners[0, 1]), zorder=1)
    ax2.set_title("Top-Down View with Scores")
    ax2.set_xlim(bounds[0][0], bounds[0][1])
    ax2.set_ylim(bounds[1][0], bounds[1][1])
    ax2.set_aspect('equal')

    # Overlay patches for each pose
    if len(scores) > 0:
        for pose, score in zip(poses, scores):
            if score > 0:  # Positive scores
                color = viridis_cmap(norm(score))
            elif score == -1:  # Object not detected
                color = color_negative_one
            elif score == -2:  # Collision check failed
                color = color_negative_two
            elif score == -3:
                color = color_negative_three
            else:
                continue

            if render_heading:
                plot_pose(ax2, pose, vis_size, color, 0.8 if score > 0 else 0.3, True)
            else:
                rect = patches.Rectangle(
                    (pose[0] - vis_size / 2, pose[1] - vis_size / 2),
                    vis_size, vis_size,
                    linewidth=0,
                    edgecolor='none',
                    facecolor=color,
                    alpha=0.8 if score > 0 else 0.3
                )
                ax2.add_patch(rect)

    # Add label for default pose
    if default_pose is not None:
        plot_pose(ax2, default_pose, vis_size * 1.5, "cyan", 1.0, True)
    if init_pose is not None:
        plot_pose(ax2, init_pose, vis_size * 1.5, "green", 1.0, True)
    if selected_pose is not None:
        plot_pose(ax2, selected_pose, vis_size * 1.5, "blue", 1.0, True)
    if nav_pose is not None:
        plot_pose(ax2, nav_pose, vis_size * 1.0, "blueviolet", 1.0, True)

    # Add colorbar for positive scores
    if len(scores) > 0:
        sm = cm.ScalarMappable(cmap=viridis_cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2, orientation='vertical', shrink=0.8)
        cbar.set_label("Raw Scores")

    # Add legend for -1 and -2 scores
    legend_patches = [
        patches.Patch(color=color_negative_one, label="Object Not Detected"),
        patches.Patch(color=color_negative_two, label="Collision Check Failed"),
        patches.Patch(color=color_negative_three, label="Out of Sight"),
        patches.Patch(color="cyan", label="Default Pose"),
        patches.Patch(color="green", label="Init Pose"),
        patches.Patch(color="blue", label="Selected Pose"),
        patches.Patch(color="blueviolet", label="Nav Pose")
    ]
    ax2.legend(handles=legend_patches, loc="lower right", title="Special Cases")

    # Save the plot
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if return_fig:
        return fig
    else:
        plt.close(fig)


def plot_best_metrics(optimal_pose, sampled_poses, scores, save_path="plot.pdf"):
    """
    Plots the best distance so far and best score so far from sampled poses and scores.

    Parameters:
        optimal_pose (np.ndarray): A 3D vector representing the optimal pose (shape: (3,)).
        sampled_poses (np.ndarray): A numpy array of sampled poses (shape: (N, 3)).
        scores (np.ndarray): A numpy array of scores for the sampled poses (shape: (N,)).
        save_path (str): Save path for the plot.
    """
    plot_data = dict(
        optimal_pose=optimal_pose,
        sampled_poses=sampled_poses,
        scores=scores
    )
    np.savez(save_path.replace(".pdf", "_plot_data.npz"), **plot_data)

    # Calculate L2 distances to the optimal pose
    distances = np.linalg.norm(sampled_poses - optimal_pose, axis=1)

    # Calculate best-so-far metrics
    best_distances_so_far = np.minimum.accumulate(distances)
    best_scores_so_far = np.maximum.accumulate(scores)

    # Plotting
    plt.figure(figsize=(12, 5))

    # Left plot: Best distance so far
    plt.subplot(1, 2, 1)
    plt.plot(best_distances_so_far, label='Best Distance So Far', color='blue')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance (L2)')
    plt.title('Best Distance So Far')
    plt.legend()
    plt.grid(True)

    # Right plot: Best score so far
    plt.subplot(1, 2, 2)
    plt.plot(best_scores_so_far, label='Best Score So Far', color='green')
    plt.xlabel('Sample Index')
    plt.ylabel('Score')
    plt.title('Best Score So Far')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)


def plot_pose_histograms(optimal_poses, selected_poses, save_path):
    """
    Plot histograms for distances between the optimal pose and a list of selected poses.

    Parameters:
        optimal_poses (numpy.ndarray): The optimal poses as (x, y, heading_rad).
        selected_poses (numpy.ndarray): A list of poses with shape (N, 3), where each pose is (x, y, heading_rad).
        save_path (str): Path to save the plot.
    """
    # Compute distances
    x_dists = np.abs(selected_poses[:, 0] - optimal_poses[:, 0])
    y_dists = np.abs(selected_poses[:, 1] - optimal_poses[:, 1])
    heading_dists = np.abs(selected_poses[:, 2] - optimal_poses[:, 2]) % (2 * np.pi)
    heading_dists = np.minimum(heading_dists, 2 * np.pi - heading_dists)  # Wrap heading distances
    xy_dists = np.sqrt((selected_poses[:, 0] - optimal_poses[:, 0])**2 + (selected_poses[:, 1] - optimal_poses[:, 1])**2)
    full_dists = np.sqrt(
        (selected_poses[:, 0] - optimal_poses[:, 0])**2 +
        (selected_poses[:, 1] - optimal_poses[:, 1])**2 +
        heading_dists**2
    )

    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))  # 3 rows, 2 columns
    axes = axes.flatten()

    # Plot histograms on separate axes
    axes[0].hist(x_dists, bins=20, alpha=0.7, color='blue')
    axes[0].set_title('Absolute X Distance')
    axes[0].set_xlabel('Distance')
    axes[0].set_ylabel('Frequency')

    axes[1].hist(y_dists, bins=20, alpha=0.7, color='green')
    axes[1].set_title('Absolute Y Distance')
    axes[1].set_xlabel('Distance')
    axes[1].set_ylabel('Frequency')

    axes[2].hist(heading_dists, bins=20, alpha=0.7, color='orange')
    axes[2].set_title('Absolute Heading Distance')
    axes[2].set_xlabel('Distance')
    axes[2].set_ylabel('Frequency')

    axes[3].hist(xy_dists, bins=20, alpha=0.7, color='red')
    axes[3].set_title('L2 XY Distance')
    axes[3].set_xlabel('Distance')
    axes[3].set_ylabel('Frequency')

    axes[4].hist(full_dists, bins=20, alpha=0.7, color='purple')
    axes[4].set_title('L2 X, Y, Heading Distance')
    axes[4].set_xlabel('Distance')
    axes[4].set_ylabel('Frequency')

    # Hide the last subplot if unused
    axes[5].axis('off')

    # Adjust layout
    fig.suptitle('Histograms of Pose Distances', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Add space for the main title
    plt.savefig(save_path)
    plt.close()


def plot_best_renders(rendered_images, sampled_scores, save_path):
    """
    Plot the best, worst, and random views based on scores and save the plot as a PDF.

    Args:
        rendered_images (np.ndarray): Array of shape (N, num_views, H, W, 3), containing rendered images.
        sampled_scores (np.ndarray): Array of shape (N,), containing scores for the sampled points.
        save_path (str): Path to save the output PDF.

    Returns:
        None
    """
    N, num_views, H, W, C = rendered_images.shape

    # Find indices of the best, worst, and random samples
    best_indices = np.argsort(sampled_scores)[-5:][::-1]  # Top 5 scores
    worst_indices = np.argsort(sampled_scores)[:5]        # Bottom 5 scores
    random_indices = np.random.choice(N, size=10, replace=False)  # 10 random samples

    # Create a grid of images (4 rows, 5 columns)
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()

    # Plot the images for the best, worst, and random views
    all_indices = list(best_indices) + list(worst_indices) + list(random_indices)
    row_labels = ['Best', 'Worst', 'Random']
    row_start = [0, 5, 15]  # Start index for each row in the grid

    for i, idx in enumerate(all_indices):
        row = next((r for r, start in enumerate(row_start) if i >= start and i < start + 5), None)
        if row is not None:
            title = f"{row_labels[row]} {i - row_start[row] + 1}"
        else:
            title = f"Random {i - row_start[2] + 1}"

        for view_idx in range(num_views):
            axes[i].imshow(rendered_images[idx, view_idx])
            axes[i].set_title(f"{title} | Image subset {idx},{view_idx}")

    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.tight_layout()

    # Save the output plot to a PDF
    plt.savefig(save_path, format='pdf')
    plt.close(fig)
