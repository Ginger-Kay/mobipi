"""
Utilities related to sim environments.

@yjy0625
"""
import os
import torch
import numpy as np
from matplotlib import pyplot as plt
import robosuite.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon, LineString, Point
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils


def get_metadata(config):
    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    env_meta_list = []
    shape_meta_list = []
    for dataset_cfg in config.train.data:
        dataset_path = os.path.expanduser(dataset_cfg["path"])
        ds_format = config.train.data_format
        if not os.path.exists(dataset_path):
            raise Exception("Dataset at provided path {} not found!".format(dataset_path))

        # load basic metadata from training file
        print("\n============= Loaded Environment Metadata =============")
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path, ds_format=ds_format)

        # populate language instruction for env in env_meta
        env_meta["env_lang"] = dataset_cfg.get("lang", None)

        # update env meta if applicable
        from robomimic.utils.script_utils import deep_update
        deep_update(env_meta, dataset_cfg.get("env_meta_update_dict", {}))
        deep_update(env_meta, config.experiment.env_meta_update_dict)
        env_meta_list.append(env_meta)

        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=dataset_path,
            action_keys=config.train.action_keys,
            all_obs_keys=config.all_obs_keys,
            ds_format=ds_format,
            verbose=True
        )
        shape_meta_list.append(shape_meta)

    return env_meta_list, shape_meta_list


def load_env(config, override_args):
    # NOTE: in our experiments, we perform seeding via policy only
    #       since we already have 10 different scenes/layouts for eval
    #       we do not perform various seeding for env and scene model.
    np.random.seed(0)
    torch.manual_seed(0)

    # set num workers
    torch.set_num_threads(1)

    env_meta_list, shape_meta_list = get_metadata(config)
    env_meta, shape_meta = env_meta_list[0], shape_meta_list[0]

    for (dataset_i, dataset_cfg) in enumerate(config.train.data):
        env_meta = env_meta_list[dataset_i]
        shape_meta = shape_meta_list[dataset_i]
        env_name = env_meta_list[dataset_i]["env_name"]
        horizon = dataset_cfg.get("horizon", config.experiment.rollout.horizon)
        break

    if not "seed" in override_args:
        env_meta["env_kwargs"]["seed"] = 0
    env_meta["env_kwargs"].update(override_args)

    print(f"Initializing environment {env_name} with seed {env_meta['env_kwargs']['seed']}.")
    env_kwargs = dict(
        env_meta=env_meta,
        env_name=env_name,
        render=False,
        render_offscreen=config.experiment.render_video,
        use_image_obs=shape_meta["use_images"],
    )
    env = EnvUtils.create_env_from_metadata(**env_kwargs)
    env = EnvUtils.wrap_env_from_config(env, config=config)

    return env


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
    from robocasa.models.fixtures import Floor
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

def plot_layout(occupied_spaces, floor_geometry, circle_position, circle_size, save_path):
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
    plt.savefig(save_path)


def get_env_map_and_default_robot_init_pose(config=None, override_args=None, save_dir=None, env=None):
    from robocasa.models.fixtures import Counter, Stove, Stovetop, HousingCabinet, Fridge, Floor

    if env is None:
        assert config is not None
        assert override_args is not None
        override_args_copy = override_args.copy()
        override_args_copy["place_robot_for_nav"] = False
        env = load_env(config, override_args_copy)
        unwrapped_env = env.unwrapped.env
        env.reset()
    else:
        unwrapped_env = env

    robot_size = unwrapped_env.robots[0].robot_model.base.horizontal_radius
    default_robot_pos, default_robot_ori = unwrapped_env._init_robot_pos, unwrapped_env._init_robot_ori

    # gets the occupancy of the environment and default robot init pose
    base_fixture_names = [fixture_name for fixture_name, fixture in unwrapped_env.fixtures.items() \
                          if (isinstance(fixture, Counter) \
                              or isinstance(fixture, Stove) \
                              or isinstance(fixture, Stovetop) \
                              or isinstance(fixture, HousingCabinet) \
                              or isinstance(fixture, Fridge))]
    floor_fixture_name = "floor_room"

    base_fixture_bounds_2d = [get_fixture_bounds_2d(unwrapped_env, base_fixture_name) for base_fixture_name in base_fixture_names]
    floor_fixture_bounds_2d = get_fixture_bounds_2d(unwrapped_env, floor_fixture_name)

    if save_dir is not None:
        layout_id, style_id = override_args["layout_and_style_ids"][0]
        plot_save_path = os.path.join(save_dir, f"plot_layout{layout_id}_style{style_id}.pdf")
        plot_layout(base_fixture_bounds_2d, floor_fixture_bounds_2d, default_robot_pos[:2], robot_size, plot_save_path)

    return base_fixture_bounds_2d, floor_fixture_bounds_2d, default_robot_pos, default_robot_ori, robot_size


def sample_robot_positions(base_fixture_bounds_2d, floor_fixture_bounds_2d, default_robot_pos, robot_size, rng, num_samples=100):
    """
    Sample valid initial robot positions on the map.

    Args:
        base_fixture_bounds_2d (list of np.ndarray): List of rectangular occupied areas.
        floor_fixture_bounds_2d (np.ndarray): Floor area defined by 4 corner points.
        default_robot_pos (np.ndarray): Default robot position as a 2D array.
        robot_size (float): Scalar representing the robot size.
        num_samples (int): Number of valid positions to sample.

    Returns:
        list: Valid robot positions as a list of 2D numpy arrays.
    """
    # Convert occupied areas and floor to Shapely Polygons
    occupied_polygons = [Polygon(bounds) for bounds in base_fixture_bounds_2d]
    floor_polygon = Polygon(floor_fixture_bounds_2d)

    # Buffer the occupied areas by robot_size to account for proximity constraint
    buffered_polygons = [polygon.buffer(robot_size) for polygon in occupied_polygons]

    # Generate random points within the floor bounds
    min_x, min_y, max_x, max_y = floor_polygon.bounds
    valid_positions = []

    num_failed_samples = 0
    while len(valid_positions) < num_samples:
        if num_failed_samples == 5000:
            # sampling not going well; try reducing robot size buffer
            buffered_polygons = [polygon.buffer(robot_size / 2) for polygon in occupied_polygons]
        if num_failed_samples == 10000:
            print(f"Warning: robot pose sampling might be stuck.")

        # Sample a random point within the floor bounds
        x = rng.uniform(min_x, max_x)
        y = rng.uniform(min_y, max_y)
        point = Point(x, y)

        # Check if the point is within the floor and not in any buffered occupied area
        if floor_polygon.contains(point) and not any(polygon.contains(point) for polygon in buffered_polygons):
            # Check the line segment from the sampled point to the default position
            line = LineString([point.coords[0], tuple(default_robot_pos)])
            if not any(polygon.intersects(line) for polygon in buffered_polygons):
                valid_positions.append(np.array([x, y]))
            else:
                num_failed_samples += 1
                # print(f"Sampled point {point.coords[0]} does not directly go towards init robot position")
        else:
            num_failed_samples += 1
            # print(f"Sampled point {point.coords[0]} within an occupied area")

    return valid_positions


def check_robot_positions(positions, base_fixture_bounds_2d, floor_fixture_bounds_2d, robot_size):
    # Convert occupied areas and floor to Shapely Polygons
    occupied_polygons = [Polygon(bounds) for bounds in base_fixture_bounds_2d]
    floor_polygon = Polygon(floor_fixture_bounds_2d).buffer(-robot_size * 2)

    # Buffer the occupied areas by robot_size to account for proximity constraint
    buffered_polygons = [polygon for polygon in occupied_polygons] # polygon.buffer(robot_size)

    # Generate random points within the floor bounds
    min_x, min_y, max_x, max_y = floor_polygon.bounds

    is_valid = []
    for position in positions:
        point = Point(position[0], position[1])
        if floor_polygon.contains(point) and not any(polygon.contains(point) for polygon in buffered_polygons):
            is_valid.append(1)
        else:
            is_valid.append(0)
    return np.array(is_valid)


def adjust_initial_robot_position(base_fixture_bounds_2d, default_robot_pos, robot_size):
    """
    Adjust the initial robot position to ensure it is outside the buffer zone of obstacles.

    Args:
        base_fixture_bounds_2d (list of np.ndarray): List of rectangular occupied areas.
        default_robot_pos (np.ndarray): Initial robot position as a 2D array.
        robot_size (float): Scalar representing the robot size.

    Returns:
        np.ndarray: Updated robot position as a 2D numpy array.
    """
    # Convert occupied areas to Shapely Polygons
    occupied_polygons = [Polygon(bounds) for bounds in base_fixture_bounds_2d]

    # Buffer the occupied areas by robot_size to account for proximity constraint
    buffered_polygons = [polygon.buffer(robot_size) for polygon in occupied_polygons]

    # Convert default_robot_pos to a Shapely Point
    robot_point = Point(default_robot_pos)

    for buffered_polygon in buffered_polygons:
        if buffered_polygon.contains(robot_point):
            # Find the closest point on the polygon boundary
            closest_point = buffered_polygon.exterior.interpolate(buffered_polygon.exterior.project(robot_point))
            closest_line = LineString(buffered_polygon.exterior.coords)

            # Compute the vector from the closest point to the robot position
            vector = np.array(closest_point.coords[0]) - np.array(robot_point.coords[0])
            dist = np.linalg.norm(vector)
            unit_vector = vector / dist

            # Back out the position to be outside the buffer zone
            adjusted_position = np.array(closest_point.coords[0]) + unit_vector * (dist + 0.01)
            return adjusted_position

    # If not within any buffer zone, return the original position
    return default_robot_pos


def sample_robot_orientations(robot_positions, default_robot_pos, rng, fov):
    """
    Sample a valid orientation for each robot position such that the default robot position is within the field of view (FOV).

    Args:
        robot_positions (list of np.ndarray): List of robot positions as 2D arrays.
        default_robot_pos (np.ndarray): Default robot position as a 2D array.
        fov (float): Field of view in radians (centered around the robot's forward direction).

    Returns:
        list: List of sampled orientations (in radians) for each robot position.
    """
    orientations = []

    for robot_pos in robot_positions:
        # Compute the angle from the robot position to the default robot position
        delta = default_robot_pos - robot_pos
        angle_to_default = np.arctan2(delta[1], delta[0])

        # Sample an orientation such that the default robot position is within the FOV
        min_angle = angle_to_default - fov / 2
        max_angle = angle_to_default + fov / 2
        orientation = rng.uniform(min_angle, max_angle)
        orientations.append(orientation)

    return orientations



def compute_robot_angle_from_mat(mat):
    # Compute the yaw (angle around Z-axis) from the quaternion
    ang = np.arctan2(-mat[0, 1], mat[0, 0])
    return ang


def compute_relative_cam_pose(env, camera_name="robot0_agentview_right"):
    robot_base_position_world, robot_base_orientation_world = env.robots[0].composite_controller.get_controller_base_pose("right")
    robot_base_position_world[2] = 0

    camera_position_world = env.sim.data.get_camera_xpos(camera_name)
    camera_orientation_world = env.sim.data.get_camera_xmat(camera_name).reshape(3, 3)

    # print(f"robot base pos: {robot_base_position_world}; robot base orientation: {robot_base_orientation_world}")
    # print(f"cam position: {camera_position_world}; cam orientation: {camera_orientation_world}")

    # Compute the relative position
    relative_position = np.dot(
        robot_base_orientation_world.T,
        (camera_position_world - robot_base_position_world)
    )

    # Compute the relative orientation
    relative_orientation = np.dot(
        robot_base_orientation_world.T,
        camera_orientation_world
    )

    return relative_position, relative_orientation


def compute_camera_extrinsics(robot_pose, relative_cam_position, relative_cam_orientation):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # robot_pose is a tensor of shape (3,) - [x, y, facing_angle]
    # relative_cam_position is a tensor of shape (3,) - Camera position relative to the robot
    # relative_cam_orientation is a tensor of shape (3, 3) - Camera orientation relative to the robot
    assert relative_cam_position.shape == torch.Size([3])
    assert relative_cam_orientation.shape == torch.Size([3, 3])

    # Unpack the robot's pose
    x, y, a = robot_pose

    # Convert the robot's facing angle 'a' to a rotation matrix (around the z-axis)
    rotation_matrix = torch.stack([
        torch.stack([torch.cos(a), -torch.sin(a), torch.tensor(0.0, device=device, dtype=robot_pose.dtype)], dim=0),
        torch.stack([torch.sin(a), torch.cos(a), torch.tensor(0.0, device=device, dtype=robot_pose.dtype)], dim=0),
        torch.tensor([0.0, 0.0, 1.0], device=device, dtype=robot_pose.dtype)
    ], dim=0)

    # Compute the camera's position in world coordinates
    camera_position_world = rotation_matrix @ relative_cam_position + torch.stack([x, y, torch.tensor(0.0, device=device, dtype=robot_pose.dtype)])

    # Compute the camera's orientation in world coordinates
    camera_orientation_world = rotation_matrix @ relative_cam_orientation

    # print(f"recovered cam position: {camera_position_world}; recovered camera orientation: {camera_orientation_world}")

    # Create the camera extrinsics (4x4 matrix)
    extrinsics = torch.eye(4, device=device)
    extrinsics[:3, :3] = camera_orientation_world
    extrinsics[:3, 3] = camera_position_world

    return extrinsics


def render_image_with_robot_mask(env, image_height=224, image_width=224, camera_name="robot0_agentview_right"):
    rel_pos, rel_mat = compute_relative_cam_pose(env.unwrapped.env, camera_name=camera_name)

    rendered_image = env.unwrapped.env.sim.render(
        camera_name=camera_name,
        width=image_width,  # Image width
        height=image_height,  # Image height
    )
    rendered_seg = env.unwrapped.env.sim.render(
        camera_name=camera_name,
        width=image_width,  # Image width
        height=image_height,  # Image height
        depth=False,  # No depth information
        segmentation=True,  # Enable segmentation
    )

    robot_geoms = [
        (geom_id, geom_name)
        for geom_id, geom_name in enumerate(env.unwrapped.env.sim.model.geom_names)
        if "robot0" in geom_name
    ]
    robot_geom_ids = [geom_id for geom_id, geom_name in robot_geoms]
    robot_mask = rendered_seg[..., 1].copy()
    robot_mask[~np.isin(rendered_seg[..., 1], robot_geom_ids)] = 0  # Set non-robot pixels to 0
    robot_mask[np.isin(rendered_seg[..., 1], robot_geom_ids)] = 1  # Set robot pixels to 1

    # flip images as images rendered in robosuite are upside down
    rendered_image, robot_mask = rendered_image[::-1], robot_mask[::-1]

    return rel_pos, rel_mat, rendered_image, robot_mask


def world_to_ros_camera_coords(env, cam_name, point_world):
    """
    Convert a 3D world coordinate into the ROS camera coordinate frame.

    Args:
        model: The MuJoCo model.
        data: The MuJoCo data.
        cam_name: Name of the camera.
        point_world: (3,) np.array in world coordinates.

    Returns:
        (3,) np.array of the point in the ROS-style camera coordinate frame.
    """
    # Get camera ID
    cam_id = env.sim.model.camera_name2id(cam_name)

    # Get camera pose in world coordinates
    cam_pos = env.sim.data.cam_xpos[cam_id]
    cam_mat = np.array(env.sim.data.cam_xmat[cam_id]).reshape(3, 3)

    # Transform world point to MuJoCo camera frame
    point_rel = point_world - cam_pos
    point_mujoco_cam = cam_mat.T @ point_rel  # In camera frame (OpenGL convention)

    # Convert from MuJoCo camera frame to ROS camera frame
    # MuJoCo: +X right, +Y up, +Z backward (looking along -Z)
    # ROS:    +X right, +Y down, +Z forward
    # So we need to flip Y and Z axes
    point_ros_cam = np.array([
        point_mujoco_cam[0],
        -point_mujoco_cam[1],
        -point_mujoco_cam[2]
    ])

    return point_ros_cam
