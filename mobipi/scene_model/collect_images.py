"""
Collects images from different views of one specific simulated scene and train
a 3DGS model.

Usage:
    python collect_images.py --env_name [environment name] \
        --layout_id [layout ID] --style_id [style ID] \
        --num_views [number of views] --out_dir [output directory]

@yjy0625
"""
import os
import re
import json
import time
import click
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, Point
from scipy.spatial.transform import Rotation as R
import open3d as o3d

from robosuite.utils.mjmod import CameraModder
from robosuite.utils.camera_utils import get_real_depth_map
import robosuite.utils.transform_utils as T

from robocasa.models.fixtures import Counter, Stove, Stovetop, HousingCabinet, Fridge, Floor
from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS, MULTI_STAGE_TASK_DATASETS
from robocasa.utils.dataset_registry import get_ds_path
from robocasa.utils.env_utils import create_env, run_random_rollouts


def camel_to_snake(name):
    """
    Convert CamelCase to snake_case.

    Args:
        name (str): The CamelCase string.

    Returns:
        str: The snake_case string.
    """
    # Add underscores between lowercase and uppercase letters, then lowercase the string
    snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    return snake_case


def apply_rotation(size, position, quat):
    r = R.from_quat(np.array(quat)[[1, 2, 3, 0]])  # Convert quaternion (w, x, y, z) to rotation matrix
    sx, sy, sz = size
    x, y, z = position

    # Define vertices of a centered cube
    vertices = np.array([
        [-sx/2, -sy/2, -sz/2], [sx/2, -sy/2, -sz/2],
        [sx/2, sy/2, -sz/2], [-sx/2, sy/2, -sz/2],
        [-sx/2, -sy/2, sz/2], [sx/2, -sy/2, sz/2],
        [sx/2, sy/2, sz/2], [-sx/2, sy/2, sz/2]
    ])

    # Apply rotation and translate vertices to object position
    rotated_vertices = r.apply(vertices) + np.array([x, y, z])
    return rotated_vertices


def get_fixture_bounds_2d(env, fixture_name):
    fixture = env.fixtures[fixture_name]
    pos = fixture.pos
    if hasattr(fixture, "get_quat"):
        quat = fixture.get_quat()
    else:
        quat = np.array([np.cos(fixture.rot / 2), 0, 0, np.sin(fixture.rot / 2)])
    if isinstance(fixture, Floor):
        size = fixture.get_bounding_box_size()
    else:
        size = fixture.size

    return apply_rotation(size, pos, quat)[:4, :2]


def plot_layout(occupied_spaces, floor_geometry, circle_position, circle_size):
    fig, ax = plt.subplots()

    # Plot floor geometry
    floor_polygon = plt.Polygon(floor_geometry, fill=None, edgecolor='black', linewidth=2)
    ax.add_patch(floor_polygon)

    # Plot occupied spaces
    for space in occupied_spaces:
        space_polygon = plt.Polygon(space, color='gray', alpha=0.6)
        ax.add_patch(space_polygon)

    # Plot circle
    circle = plt.Circle(circle_position, circle_size, color='blue', alpha=0.4)
    ax.add_patch(circle)

    ax.set_aspect('equal', 'box')
    ax.set_xlim([floor_geometry[:, 0].min() - circle_size, floor_geometry[:, 0].max() + circle_size])
    ax.set_ylim([floor_geometry[:, 1].min() - circle_size, floor_geometry[:, 1].max() + circle_size])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Floor Plan Layout with Circle")
    plt.show()


def is_collision_free(occupied_spaces, floor_geometry, circle_position, circle_size):
    # Create the floor polygon
    floor_polygon = Polygon(floor_geometry)

    # Check if the circle is contained within the floor polygon
    circle = Point(circle_position).buffer(circle_size)
    if not floor_polygon.contains(circle):
        return False  # Circle is out of bounds

    # Check for collision with each occupied space
    for space in occupied_spaces:
        space_polygon = Polygon(space)
        if space_polygon.intersects(circle):
            return False  # Circle collides with an occupied space

    return True  # Circle is collision-free


def sample_valid_robot_position(occupied_spaces, floor_geometry, circle_size, max_attempts=1000):
    # Create the floor polygon
    floor_polygon = Polygon(floor_geometry)

    # Determine the bounding box for the floor geometry
    min_x, min_y, max_x, max_y = floor_polygon.bounds

    # Attempt to sample a valid position up to `max_attempts` times
    for _ in range(max_attempts):
        # Randomly sample a position within the bounding box
        x = random.uniform(min_x + circle_size, max_x - circle_size)
        y = random.uniform(min_y + circle_size, max_y - circle_size)
        candidate_position = (x, y)
        
        # Check if the sampled position is valid (collision-free and within bounds)
        if is_collision_free(occupied_spaces, floor_geometry, candidate_position, circle_size):
            return candidate_position  # Return the first valid position found
    
    # If no valid position is found after `max_attempts`, return None
    print("No valid position found after maximum attempts.")
    return None


def render_image(env, camera_name, camera_position, camera_quat, width=512, height=512, hide_robot=True, fovy=90):
    camera_modder = CameraModder(env.sim)
    camera_modder.set_pos(camera_name, camera_position)
    camera_modder.set_quat(camera_name, camera_quat)
    camera_modder.set_fovy(camera_name, fovy)

    if hide_robot:
        for i in range(env.sim.model.ngeom):
            geom_name = env.sim.model.geom_id2name(i)
            if "robot0" in geom_name or "base0" in geom_name or "gripper0" in geom_name:
                env.sim.model.geom_rgba[i, 3] = 0  # Set alpha to 0 for invisibility

    env.sim.forward()
    image, depth = env.sim.render(width=width, height=height, camera_name=camera_name, depth=True)

    return image[::-1], get_real_depth_map(env.sim, depth[::-1])


def save_nerfstudio_data(data, output_dir):
    """
    Saves images and computes camera intrinsics for a Nerfstudio-compatible dataset.

    Args:
        data (dict): A dictionary with keys:
            - "robot_pos": Unused for now.
            - "cam_pos": List of camera positions (list of lists, each of shape (3,)).
            - "cam_xmat": List of camera matrices (list of matrices, each of shape (3,3)).
            - "images": List of image arrays (H, W, 3).
            - "cam_fovy": List of vertical field of view values in radians (one per frame).
        output_dir (str): Path to the directory where the data will be saved.

    Returns:
        None
    """
    # Extract image dimensions
    h, w = data["image_size"], data["image_size"]

    # Compute intrinsics from fovy
    # Assumes square pixels, so fl_x = fl_y
    # fov = 2 * Math.atan( (h/2) / fl_y)) * 180 / pi;
    # fl_y = (h/2) / (tan(fov_radian / 2))
    cam_fovy = np.mean(data["cam_fovy"]) / 180 * np.pi  # Average FOV across all frames
    fl_y = h / (2 * np.tan(cam_fovy / 2))
    fl_x = fl_y  # Assuming square pixels

    # Principal point (image center)
    cx = w / 2
    cy = h / 2

    # Prepare transform data
    frames = []

    for i, (pos, mat) in tqdm(enumerate(zip(data["cam_pos"], data["cam_xmat"]))):
        # Create transform matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = mat
        transform_matrix[:3, 3] = pos

        # Append frame metadata
        frames.append({
            "file_path": f"./images/frame_{i:04d}.png",
            "transform_matrix": transform_matrix.tolist()
        })

    # Create the global dictionary for transforms.json
    transforms = {
        "camera_model": "OPENCV",
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "ply_file_path": os.path.join(output_dir, "pc.ply"),
        "frames": frames
    }

    # Save the transforms.json
    transforms_path = os.path.join(output_dir, "transforms.json")
    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=4)

    print(f"Nerfstudio data saved to {output_dir}")


def transform_pixels2world(pixels, rgb, depth, cam_mat):
    # pixels: N, 2
    # rgb: H, W, 3
    # depth: H, W
    # cam_mat: 4, 4
    pixels = pixels.astype(int)
    z = depth[pixels[:, 0], pixels[:, 1]]
    z = z.reshape(-1, 1)
    rgb_list = rgb[pixels[:, 0], pixels[:, 1]]

    cam_pts = [pixels[..., 1:2] * z, pixels[..., 0:1] * z, z, np.ones_like(z)]
    cam_pts = np.concatenate(cam_pts, axis=-1)

    mat_reshape = [1] * len(cam_pts.shape[:-1]) + [4, 4]
    cam_trans = cam_mat.reshape(mat_reshape)
    points = np.matmul(cam_trans, cam_pts[..., None])[..., 0]
    return np.concatenate([points[..., :3], rgb_list], axis=1)


def unproject_depth_to_world(depth, rgb, cam_intrinsics, cam_extrinsics):
    """
    Unproject depth into a 3D point cloud in world coordinates with associated RGB colors.

    Args:
        depth (np.ndarray): Depth image of shape (H, W).
        rgb (np.ndarray): Corresponding RGB image of shape (H, W, 3).
        cam_intrinsics (np.ndarray): Camera intrinsics matrix of shape (3, 3).
        cam_extrinsics (np.ndarray): Camera extrinsics matrix of shape (4, 4) (cam-to-world transformation).
                                     This should include both rotation and translation.

    Returns:
        np.ndarray: 3D points of shape (N, 6), where each row is (x, y, z, r, g, b) in world coordinates.
    """
    H, W = depth.shape
    fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
    cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]

    # Create a grid of (x, y) pixel coordinates
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = depth.flatten()

    # Filter out invalid depth values
    valid = np.logical_and(z > 0, z < 50)
    x, y, z = x[valid], y[valid], z[valid]

    # Unproject to camera coordinates
    x_cam = (x - cx) * z / fx
    y_cam = (y - cy) * z / fy
    points_cam = np.vstack((x_cam, y_cam, z, np.ones_like(z))).T  # Homogeneous coordinates (N, 4)

    # Transform to world coordinates using the extrinsics
    points_world_hom = points_cam @ cam_extrinsics.T

    # Extract 3D points from homogeneous coordinates
    points_world = points_world_hom[:, :3]

    # Add corresponding RGB colors
    rgb_flat = rgb.reshape(-1, 3)[valid]
    point_cloud_world = np.hstack((points_world, rgb_flat))  # Combine points with RGB

    return point_cloud_world


def sample_point_cloud(point_cloud, K):
    """
    Randomly sample K points from the point cloud.

    Args:
        point_cloud (np.ndarray): Input point cloud of shape (N, 6).
        K (int): Number of points to sample.

    Returns:
        np.ndarray: Sampled point cloud of shape (K, 6).
    """
    if point_cloud.shape[0] <= K:
        return point_cloud
    indices = np.random.choice(point_cloud.shape[0], K, replace=False)
    return point_cloud[indices]


def get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width):
    """
    Obtains camera intrinsic matrix.

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
    Return:
        K (np.array): 3x3 camera matrix
    """
    cam = sim.model.camera(camera_name)
    fovy = cam.fovy[0]
    f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
    K = np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])
    return K


def get_camera_extrinsic_matrix(sim, camera_name):
    """
    Returns a 4x4 homogenous matrix corresponding to the camera pose in the
    world frame. MuJoCo has a weird convention for how it sets up the
    camera body axis, so we also apply a correction so that the x and y
    axis are along the camera view and the z axis points along the
    viewpoint.
    Normal camera convention: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
    Return:
        R (np.array): 4x4 camera extrinsic matrix
    """
    cam = sim.data.camera(camera_name)
    camera_pos = cam.xpos
    camera_rot = cam.xmat.reshape(3, 3)
    R = T.make_pose(camera_pos, camera_rot)

    # IMPORTANT! This is a correction so that the camera axis is set up along the viewpoint correctly.
    camera_axis_correction = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    R = R @ camera_axis_correction
    return R


def get_camera_transform_matrix(sim, camera_name, camera_height, camera_width):
    """
    Camera transform matrix to project from world coordinates to pixel coordinates.

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
    Return:
        K (np.array): 4x4 camera matrix to project from world coordinates to pixel coordinates
    """
    R = get_camera_extrinsic_matrix(sim=sim, camera_name=camera_name)
    K = get_camera_intrinsic_matrix(
        sim=sim,
        camera_name=camera_name,
        camera_height=camera_height,
        camera_width=camera_width,
    )
    K_exp = np.eye(4)
    K_exp[:3, :3] = K

    # Takes a point in world, transforms to camera frame, and then projects onto image plane.
    return K_exp @ T.pose_inv(R)


@click.command()
@click.option("--env_name", default="TurnOnSinkFaucet", type=str)
@click.option("--seed", default=0, type=int)
@click.option("--layout_id", default=1, type=int)
@click.option("--style_id", default=1, type=int)
@click.option("--num_views", default=1000, type=int)
@click.option("--out_dir", default="scene_data")
@click.option("--pointcloud_samples", default=250000, type=int, help="Number of points to sample from the point cloud.")
@click.option("--image_size", default=512, type=int)
@click.option("--fovy", default=60, type=int)
def main(env_name, seed, layout_id, style_id, num_views, out_dir, pointcloud_samples, image_size, fovy):
    start_time = time.time()

    # Create directories
    data_dir = os.path.join(out_dir, f"{camel_to_snake(env_name)}/layout{layout_id}_style{style_id}_seed{seed}")
    image_dir = os.path.join(data_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    env_names = list(SINGLE_STAGE_TASK_DATASETS) + list(MULTI_STAGE_TASK_DATASETS)
    env = create_env(env_name=env_name, seed=seed, layout_ids=layout_id, style_ids=style_id, place_robot=False,
                     obj_instance_split="B")
    obs = env.reset()

    camera_name = "freeview"

    print(f"Camera names: {list(env.sim.model.camera_names)}")
    
    base_fixture_names = [fixture_name for fixture_name, fixture in env.fixtures.items() \
                          if (isinstance(fixture, Counter) \
                              or isinstance(fixture, Stove) \
                              or isinstance(fixture, Stovetop) \
                              or isinstance(fixture, HousingCabinet) \
                              or isinstance(fixture, Fridge))]
    floor_fixture_name = "floor_room"
    
    base_fixture_bounds_2d = [get_fixture_bounds_2d(env, base_fixture_name) for base_fixture_name in base_fixture_names]
    floor_fixture_bounds_2d = get_fixture_bounds_2d(env, floor_fixture_name)
    
    robot_size = env.robots[0].robot_model.base.horizontal_radius
    
    full_point_cloud = []  # Initialize an empty list for the full point cloud

    data = {"robot_pos": [], "cam_pos": [], "cam_xmat": [], "images": [], "cam_fovy": [], "image_size": image_size}
    data_idx = 0
    for i in tqdm(range(num_views // 4), desc="Generating views"):
        sampled_pos = sample_valid_robot_position(base_fixture_bounds_2d, floor_fixture_bounds_2d, circle_size=robot_size)
        assert is_collision_free(base_fixture_bounds_2d, floor_fixture_bounds_2d, sampled_pos, circle_size=robot_size)
        sampled_heading = np.random.rand() * np.pi * 2
        for height, euler_ang in zip([3.0, 2.1, 1.2, 0.3], [60, 75, 105, 120]):
            desired_cam_pos = np.array([sampled_pos[0], sampled_pos[1], height])
            desired_cam_euler = np.array([euler_ang / 180 * np.pi, 0, sampled_heading])
            desired_cam_quat = R.from_euler('xyz', desired_cam_euler).as_quat()[[3, 0, 1, 2]]

            # Render RGB and depth
            image, depth = render_image(env, camera_name, desired_cam_pos, desired_cam_quat,
                                        width=image_size, height=image_size, fovy=fovy)
            image_save_path = os.path.join(image_dir, f"frame_{data_idx:04d}.png")
            Image.fromarray(image).save(image_save_path)

            cam_id = env.sim.model.camera_name2id(camera_name)
            assert cam_id >= 0
            cam_pos = env.sim.data.cam_xpos[cam_id]
            cam_xmat = np.array(env.sim.data.cam_xmat[cam_id]).reshape(3, 3)
            cam_quat = R.from_matrix(cam_xmat).as_quat()[[3, 0, 1, 2]]
            cam_world2pix = get_camera_transform_matrix(
                env.sim, camera_name, image_size, image_size
            )
            cam_pix2world = np.linalg.inv(cam_world2pix)
            pixels = np.argwhere(depth < 50)

            assert np.allclose(cam_pos, desired_cam_pos, atol=1e-3)
            assert np.allclose(cam_quat, desired_cam_quat, atol=1e-3) or np.allclose(cam_quat, -desired_cam_quat, atol=1e-3)

            # Unproject to point cloud
            point_cloud = sample_point_cloud(transform_pixels2world(pixels, image, depth, cam_pix2world), 8192)
            full_point_cloud.append(point_cloud)  # Append to the full point cloud

            data["robot_pos"].append(sampled_pos)
            data["cam_pos"].append(cam_pos.copy())
            data["cam_xmat"].append(cam_xmat.copy())
            data["cam_fovy"].append(env.sim.model.cam_fovy[cam_id])

            data_idx += 1

    # Merge all sampled point clouds into one and save as a single PLY file
    full_point_cloud = sample_point_cloud(np.vstack(full_point_cloud), pointcloud_samples * 4)
    print(f"Got a full point cloud with shape {full_point_cloud.shape}")
    ply_file = os.path.join(data_dir, "pc.ply")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(full_point_cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(full_point_cloud[:, 3:] / 255.0)
    print(f"Created point cloud object")
    if len(pcd.points) > pointcloud_samples:
        pcd = pcd.voxel_down_sample(voxel_size=0.02)
        print(f"Performed point cloud sampling.")
    o3d.io.write_point_cloud(ply_file, pcd)
    print(f"Full point cloud saved to {ply_file}")

    save_nerfstudio_data(data, data_dir)

    del data
    del pcd
    del full_point_cloud

    print(f"Scene data generation took {time.time() - start_time:.2f}s")
    os.system(f'ns-train splatfacto-big --data {data_dir} --max-num-iterations 30000 --output-dir {os.path.join(data_dir, "model")} --project_name "" --experiment_name "" --viewer.quit-on-train-completion True --pipeline.model.background_color "white" --viewer.websocket-port $((RANDOM % 1001 + 7000))')


if __name__ == "__main__":
    main()
