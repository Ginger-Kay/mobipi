"""
Utilities related to navigation.

@yjy0625, @branvu
"""
import numpy as np
import torch
import cv2
import robosuite.utils.transform_utils as T
import mimicgen.utils.pose_utils as PoseUtils
import matplotlib.pyplot as plt
import math
import random
import time

from mobipi.utils.media_utils import to_fisheye
from mobipi.utils.env_utils import world_to_ros_camera_coords

# -------------------------------------------------------------------
#  CONFIGURABLE CONSTANTS
# -------------------------------------------------------------------
MAX_ITER = 10000  # Max number of RRT iterations
STEP_SIZE = 0.1  # Distance to move from a node toward the sample
GOAL_THRESHOLD = (
    0.5  # 0.2  # When we're "close enough" to the goal in Euclidean distance
)
ANGLE_THRESHOLD = 1.3  # When orientation is "close enough"
X_MIN, X_MAX = 0, 5  # Bounds for x
Y_MIN, Y_MAX = -6, 0  # Bounds for y
GOAL_SAMPLE_PROB = 0.1
# For theta, we might assume free rotation in [-pi, pi].


# -------------------------------------------------------------------
#  DATA STRUCTURES
# -------------------------------------------------------------------
class Node:
    """
    A node in the RRT tree.
    """

    def __init__(self, x, y, theta, parent=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent = parent  # Index of the parent node in the tree


def distance_pose(pose1, pose2):
    """
    Euclidean distance in (x, y) plus an angular difference component.
    This is one possible distance metric. Adjust if needed.
    """
    dx = pose1[0] - pose2[0]
    dy = pose1[1] - pose2[1]
    # For rotation, you could do a wrap-around difference if needed.
    dtheta = abs(pose1[2] - pose2[2])
    return math.sqrt(dx * dx + dy * dy) + dtheta


def sample_random_pose():
    """
    Sample a random (x, y, theta) within some bounding box / angle range.
    """
    x = random.uniform(X_MIN, X_MAX)
    y = random.uniform(Y_MIN, Y_MAX)
    theta = random.uniform(-math.pi, math.pi)
    return (x, y, theta)


def angle_wrap(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def steer(from_pose, to_pose, step_size=STEP_SIZE):
    """
    Move from 'from_pose' toward 'to_pose' by at most step_size in linear space
    and proportionally in rotation. Returns the "intermediate" pose.
    """
    x0, y0, th0 = from_pose
    x1, y1, th1 = to_pose

    dx = x1 - x0
    dy = y1 - y0
    dist_xy = math.sqrt(dx * dx + dy * dy)

    if dist_xy < 1e-6:
        dist_xy = 0.0

    # Fraction of the distance to move
    frac = step_size / dist_xy if dist_xy > step_size else 1.0
    # Interpolate orientation linearly
    dtheta = th1 - th0

    new_x = x0 + frac * dx
    new_y = y0 + frac * dy

    # Interpolate orientation over the shortest angular distance
    dtheta = angle_wrap(th1 - th0)
    new_theta = angle_wrap(th0 + frac * dtheta)

    return (new_x, new_y, new_theta)


def is_collision_free(k_col, poses):
    """
    Checks collision by calling your collision function k_col.compute_score.
    'poses' is an Nx3 array of [x, y, theta].
    We'll assume the function returns 0 for collision-free and >0 for collisions,
    or something similar. Adjust the check as needed.
    """
    poses_arr = np.array(poses).reshape(-1, 3)
    scores = ~k_col.compute_score(
        None, None, poses_arr, numpy=True, robot_size=(k_col.robot_size + 0.5) / 2
    )
    # If the function returns an array, we consider ANY score > 0 to be collision.
    return not np.any(scores > 0)


def check_path_collision(k_col, start_pose, end_pose, steps=10):
    """
    Check collision along the line (and interpolated angles)
    from start_pose to end_pose by discretizing into 'steps'.
    """
    path_poses = []
    for i in range(steps + 1):
        frac = i / float(steps)
        x = start_pose[0] + frac * (end_pose[0] - start_pose[0])
        y = start_pose[1] + frac * (end_pose[1] - start_pose[1])
        # Simple interpolation of theta
        th = angle_wrap(start_pose[2] + frac * angle_wrap(end_pose[2] - start_pose[2]))
        path_poses.append((x, y, th))

    return is_collision_free(k_col, path_poses)


def rrt_planner(k_col, start, goal, draw=False):
    """
    Builds an RRT from 'start' to 'goal', each is (x, y, theta).
    'k_col' is the collision-checking object.
    Returns a list of (x, y, theta) if successful, or None if not.
    """
    # Initialize tree with the start pose
    tree = [Node(start[0], start[1], start[2], parent=None)]
    best_dist = float("inf")
    best_idx = 0
    for iteration in range(MAX_ITER):
        # 1. Sample a random pose
        if np.random.rand() < GOAL_SAMPLE_PROB:
            rand_pose = goal
        else:
            rand_pose = sample_random_pose()

        # 2. Find nearest node in the tree
        nearest_idx = None
        min_dist = float("inf")
        for i, node in enumerate(tree):
            da = distance_pose((node.x, node.y, node.theta), rand_pose)

            if da < min_dist:
                min_dist = da
                nearest_idx = i

        nearest_node = tree[nearest_idx]
        nearest_pose = (nearest_node.x, nearest_node.y, nearest_node.theta)

        # 3. Steer toward that random pose
        new_pose = steer(nearest_pose, rand_pose, STEP_SIZE)

        # 4. Check collision from nearest_pose -> new_pose
        if check_path_collision(k_col, nearest_pose, new_pose):
            # It's safe to add new node
            new_node = Node(*new_pose, parent=nearest_idx)
            tree.append(new_node)

            d_goal = distance_pose(new_pose, goal)
            if d_goal < best_dist:
                best_dist = d_goal
                best_idx = len(tree) - 1

            # 5. Check if we reached (or are close to) the goal
            if (
                np.linalg.norm(np.array(new_pose[:2]) - np.array(goal[:2]))
                < GOAL_THRESHOLD
            ):
                # Check orientation closeness if it matters
                if abs(angle_wrap(new_pose[2] - goal[2])) < ANGLE_THRESHOLD:

                    path = build_solution_path(tree, len(tree) - 1, goal)

                    a = time.time()
                    path_short = shortcut_path(path, k_col, n_iterations=200, steps=200)
                    print("shortcut time", time.time() - a)
                    print(
                        f"Reached goal in {iteration} iterations! Length of path {len(path)}...shortened to {len(path_short)}"
                    )
                    if draw:
                        draw_rrt(tree, start, goal, path=path_short, title="RRT")
                    return path_short

    path = build_solution_path(
        tree, best_idx, (tree[best_idx].x, tree[best_idx].y, tree[best_idx].theta)
    )

    # "Force" the last waypoint to be the actual goal
    # NOTE: The step from path[-2] to path[-1] is not guaranteed collision-free.
    path[-1] = goal

    # Optionally do shortcut:
    path_short = shortcut_path(path, k_col, n_iterations=200, steps=200)

    # If we exit the loop, no path found
    print(
        "RRT failed to find a path within max iterations but trying best path anyways len:",
        len(path_short),
    )
    if draw:
        draw_rrt(tree, start, goal, path=path_short, title="RRT")
    # breakpoint()
    return path_short


def shortcut_path(path, k_col, n_iterations=100, steps=20):
    """
    Iterative shortcutting on an (x, y, theta) path.

    Args:
        path (list): A list of (x, y, theta) waypoints.
        k_col (object): Your collision checker, used by check_path_collision.
        n_iterations (int): How many attempts to try random shortcuts.
        steps (int): Number of interpolation steps when checking collision between two waypoints.

    Returns:
        list: A shortened/smoothed path.
    """
    if len(path) < 3:
        return path  # Nothing to shortcut if only start/goal or so

    path = list(path)  # Make a copy if needed

    for _ in range(n_iterations):
        # Pick two random indices i < j in the path
        i = random.randint(0, len(path) - 2)
        j = random.randint(i + 1, len(path) - 1)

        # If they're neighbors, no sense shortcutting
        if j <= i + 1:
            continue

        # Check collision from path[i] directly to path[j]
        if check_path_collision(k_col, path[i], path[j], steps=steps):
            # If collision-free, remove the in-between waypoints
            new_path = path[: i + 1] + path[j:]
            path = new_path

    return path


def build_solution_path(tree, goal_idx, goal_pose):
    """
    Reconstruct the path from the tree by walking from the goal node
    back to the start, then reversing the list.
    """
    path = []
    current_idx = goal_idx
    while current_idx is not None:
        node = tree[current_idx]
        path.append((node.x, node.y, node.theta))
        current_idx = node.parent

    path.reverse()
    # Optionally snap to the exact goal pose orientation if needed:
    path[-1] = goal_pose
    return path


def draw_rrt(tree, start, goal, path=None, title="RRT", save_path="my_rrt_plot.png"):
    """
    Draws (and saves) the RRT tree in 2D, ignoring orientation.

    Arguments:
      - tree:    list of Node objects
      - start:   (x, y, theta)
      - goal:    (x, y, theta)
      - path:    list of (x, y, theta) for the solution path (if found)
      - title:   string, plot title
      - save_path: file path or name to save the figure (PNG, PDF, etc.)
    """
    plt.figure(figsize=(8, 8))
    plt.title(title)

    # Plot the tree
    for i, node in enumerate(tree):
        # Node
        plt.plot(node.x, node.y, "ro", markersize=2)
        # Edge to parent
        if node.parent is not None:
            parent = tree[node.parent]
            plt.plot([node.x, parent.x], [node.y, parent.y], "k-", linewidth=0.5)

    # Mark start / goal
    plt.plot(start[0], start[1], "gs", markersize=8, label="Start")
    plt.plot(goal[0], goal[1], "bs", markersize=8, label="Goal")

    # If path exists, plot it in green
    if path is not None and len(path) > 1:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_x, path_y, "g-", linewidth=2, label="Path")
        plt.plot(path_x, path_y, "go", markersize=4)

    plt.xlim(X_MIN - 1, X_MAX + 1)
    plt.ylim(Y_MIN - 1, Y_MAX + 1)
    plt.gca().set_aspect("equal", "box")
    plt.legend()

    # Save the figure (no plt.show() so it doesn't pop up)
    plt.savefig(save_path, dpi=300)
    plt.close()  # close the figure so it doesn't remain in memory


def target_base_pose_to_action(target_base_pose, curr_base_pose, prev_base_pose,
                               dt, legacy=True):
    """
    Takes a target base pose and returns an action
    (usually a normalized delta pose action) to try and achieve that target pose.

    Args:
        target_base_pose (np.array): 4x4 target base pose
        curr_base_pose (np.array): 4x4 current base pose
        prev_base_pose (np.array): 4x4 last step base pose
        legacy (bool): if True, use legacy nav code compatible with how the benchmark results are generated.

    Returns:
        action (np.array): action compatible with env.step (base-only)
    """
    # target position and rotation
    target_pos, target_rot = PoseUtils.unmake_pose(target_base_pose)
    target_angle = T.quat2axisangle(T.mat2quat(target_rot))[2]

    # functions to convert delta poses to actions
    def delta_xy_to_action(delta, prev_delta):
        # input: 3d array indicating intended delta in xy and heading
        # output: 3d array with the same size
        if legacy:
            v = delta * 20
            return (np.abs(v) > 0.00025) * np.clip(
                v + (0.25 - 0.00025) * np.sign(v), -1, 1
            ) + (np.abs(v) <= 0.00025) * (v * 1e3)
        else:
            error = delta
            d_error = (delta - prev_delta) / dt
            kp, kd = 40.0, 0.0
            ac = error * kp + d_error * kd
            if np.linalg.norm(ac) > 1.0:
                ac /= np.linalg.norm(ac)
            return ac

    def delta_heading_to_action(delta, prev_delta):
        # Legacy code might result in oscillations around target heading.
        # Updated code fixes this.
        if legacy:
            return np.clip(delta * (20 / 0.744), -1, 1)
        else:
            error = delta
            d_error = (delta - prev_delta) / dt
            kp, kd = 4.0, 0.0
            return np.clip(error * kp + d_error * kd, -1, 1)

    curr_base_pos, curr_base_rot = PoseUtils.unmake_pose(curr_base_pose)
    prev_base_pos, prev_base_rot = PoseUtils.unmake_pose(prev_base_pose)

    # normalized delta position action
    delta_position = (target_pos - curr_base_pos)[:2]
    prev_delta_position = (target_pos - prev_base_pos)[:2]
    action_pos = delta_xy_to_action(delta_position, prev_delta_position)

    # normalized delta rotation action
    delta_rotation = T.quat2axisangle(T.mat2quat(target_rot.dot(
        curr_base_rot.T)))
    prev_delta_rotation = T.quat2axisangle(T.mat2quat(target_rot.dot(
        prev_base_rot.T)))
    action_rot = delta_heading_to_action(delta_rotation, prev_delta_rotation)
    assert np.all(action_rot[:2] == 0)

    # convert to action in base frame
    base_angle = T.quat2axisangle(T.mat2quat(curr_base_rot))[2]
    x_w, y_w = action_pos[0], action_pos[1]
    x_r = np.cos(base_angle) * x_w + np.sin(base_angle) * y_w
    y_r = -np.sin(base_angle) * x_w + np.cos(base_angle) * y_w
    action_pos[0] = x_r
    action_pos[1] = y_r

    return np.concatenate([action_pos, action_rot[[-1]]])


def compute_bounding_box(obj_mask):
    # Find the non-zero (or True) elements in the mask
    indices = np.argwhere(obj_mask)

    if indices.size == 0:
        return None  # Return None if the mask is empty

    h_min, w_min = indices.min(axis=0)
    h_max, w_max = indices.max(axis=0)

    return [h_min, h_max, w_min, w_max]


def get_object_data(env):
    obj_name = env.get_object_name()

    if hasattr(env, "microwave"):
        ref = env.microwave
    elif hasattr(env, "drawer"):
        ref = env.drawer
    elif hasattr(env, "door_fxtr"):
        ref = env.door_fxtr
    elif hasattr(env, "stove"):
        ref = env.stove
    elif hasattr(env, "sink"):
        ref = env.sink

    scores = []
    masks = []
    num_pixels = []

    # render an image at this pose
    rendered_seg = env.sim.render(
        camera_name="robot0_fisheye",
        width=1024,
        height=1024,
        depth=False,
        segmentation=True,
    )[::-1]
    obj_geoms = [
        (geom_id, geom_name)
        for geom_id, geom_name in enumerate(
            env.sim.model.geom_names
        )
        if ref.name in geom_name
    ]
    obj_geom_ids = [geom_id for geom_id, geom_name in obj_geoms]
    obj_mask = (np.isin(rendered_seg[..., 1], obj_geom_ids)).astype(int)
    obj_mask_rgb = np.stack((obj_mask,)*3, axis=-1)
    obj_mask = (to_fisheye(obj_mask_rgb.astype(np.uint8) * 255)[..., 0] > 0).astype(np.uint8)
    obj_mask = cv2.resize(obj_mask, (224, 224))

    if np.sum(obj_mask) > 0:
        bbox = compute_bounding_box(obj_mask)
        obj_pos_world = np.array(ref.get_bbox_points()).mean(0)
        obj_pos_camera = world_to_ros_camera_coords(env, "robot0_fisheye", obj_pos_world)
        obj_detect = True
    else:
        bbox = None
        obj_pos_camera = None
        obj_detect = False

    return bbox, obj_pos_camera, obj_detect


def move_to_pose(
    env,
    target_pose,
    render=False,
    camera_name="robot0_agentview_left",
    verbose=False,
    use_rrt=None,
    k_col=None,
    unwrapped=False,
    save_obj_data=False,
    step_callback=None,
    # Note:
    # Recently, we have made a few fixes to the navigation code.
    # This makes the qualitative behavior of the navigation phase more pleasing
    # but quantitatively it minimally influences the performance of our method.
    # To use the qualitatively better navigation code, turn legacy to False.
    legacy=True
):
    if not unwrapped:
        unwrapped_env = env.unwrapped.env
    else:
        unwrapped_env = env

    base_pos_history = []
    base_mat_history = []
    base_vec_history = []
    base_velp_history = []
    base_velr_history = []
    base_cmd_history = []
    base_error_history = []
    image_history = []

    if save_obj_data:
        bbox_history = []
        pose_median_history = []
        obj_detect_history = []

    robot_base_name = "mobilebase0_base"

    max_rot_per_step = 1.0 / 20
    max_dist_per_step = 0.5 / 20

    def compute_num_steps(dists):
        dists[2] = np.mod(dists[2], np.pi * 2) - np.pi
        dists = np.abs(dists)
        if legacy:
            num_steps = np.max(
                np.ceil(
                    dists
                    / np.array([max_dist_per_step, max_dist_per_step, max_rot_per_step])
                )
            )
        else:
            num_steps = max(np.linalg.norm(dists[:2]) / max_dist_per_step,
                            dists[2] / max_rot_per_step)
        return int(num_steps)

    reset_joint_pos = unwrapped_env.sim.data.qpos[
        unwrapped_env.robots[0]._ref_joint_pos_indexes
    ]

    robot_controller = unwrapped_env.robots[0].composite_controller
    init_pos, init_rot = robot_controller.get_controller_base_pose("right")
    init_pos[2] = 0
    init_pose = PoseUtils.make_pose(init_pos, init_rot)
    init_heading = T.quat2axisangle(T.mat2quat(init_rot))[2]
    init_vec = np.array([init_pos[0], init_pos[1], init_heading])

    target_pos, target_rot = PoseUtils.unmake_pose(target_pose)
    target_pos[2] = 0
    target_heading = T.quat2axisangle(T.mat2quat(target_rot))[2]
    target_vec = np.array([target_pos[0], target_pos[1], target_heading])

    target_direction = target_pos - init_pos
    nav_heading = np.arctan2(target_direction[1], target_direction[0])
    nav_rot = T.quat2mat(T.axisangle2quat(np.array([0, 0, nav_heading])))

    wp1_vec = np.array([init_pos[0], init_pos[1], nav_heading])
    wp2_vec = np.array([target_pos[0], target_pos[1], nav_heading])
    waypoints = [
        # init
        [PoseUtils.make_pose(init_pos, init_rot), init_vec],
        # rotate towards target
        [PoseUtils.make_pose(init_pos, nav_rot), wp1_vec],
        # go to target position
        [PoseUtils.make_pose(target_pos, nav_rot), wp2_vec],
        # rotate to target heading
        [PoseUtils.make_pose(target_pos, target_rot), target_vec],
    ]

    if not check_path_collision(k_col, init_vec, target_vec, steps=100):
        print("Straight line path not collision free, using RRT...")
        path = rrt_planner(
            k_col=k_col,
            start=(init_pos[0], init_pos[1], init_heading),
            goal=(target_pos[0], target_pos[1], target_heading),
            draw=False,
        )

        waypoints = []
        for point_idx, point in enumerate(path):
            waypoints.append([
                PoseUtils.make_pose(
                    np.array([point[0], point[1], 0.0]),
                    T.quat2mat(T.axisangle2quat([0, 0, point[2]])),
                ),
                point,
            ])
        waypoints.append([PoseUtils.make_pose(target_pos, target_rot), target_vec])

    # record initial observations
    base_pos, base_rot = robot_controller.get_controller_base_pose("right")
    base_heading = T.quat2axisangle(T.mat2quat(base_rot))[2]
    base_vec = np.array([base_pos[0], base_pos[1], base_heading])
    base_pos_history.append(base_pos.copy())
    base_mat_history.append(base_rot.copy())
    base_vec_history.append(base_vec)

    error = np.abs(base_vec - target_vec)
    base_error_history.append(error)

    base_velp = unwrapped_env.sim.data.get_body_xvelp(robot_base_name)
    base_velr = unwrapped_env.sim.data.get_body_xvelr(robot_base_name)
    base_velp_history.append(base_velp.copy())
    base_velr_history.append(base_velr.copy())

    if save_obj_data:
        bbox, pose_median, obj_detect = get_object_data(unwrapped_env)
        bbox_history.append(bbox)
        pose_median_history.append(pose_median)
        obj_detect_history.append(obj_detect)

    if render:
        if "fisheye" in camera_name:
            img = unwrapped_env.sim.render(width=1024, height=1024,
                                           camera_name=camera_name)[::-1]
        else:
            obs = env.get_observation()
            img = obs[f"{camera_name}_image"].transpose(1, 2, 0)
        image_history.append((img.copy() * 255).astype(np.uint8))

    if save_obj_data:
        bbox, pose_median, obj_detect = get_object_data(unwrapped_env)
        bbox_history.append(bbox)
        pose_median_history.append(pose_median)
        obj_detect_history.append(obj_detect)

    for i in range(len(waypoints) - 1):
        prev_pose = PoseUtils.make_pose(
            *robot_controller.get_controller_base_pose("right")
        )
        prev_vec = T.quat2axisangle(T.mat2quat(prev_pose[:3, :3]))[2]
        next_pose, next_vec = waypoints[i + 1]
        dt = 1.0 / unwrapped_env.control_freq

        num_steps = compute_num_steps(next_vec - prev_vec)
        if not legacy:
            stuck_counter = 0
        for t in range(num_steps):
            curr_pose = PoseUtils.make_pose(
                *robot_controller.get_controller_base_pose("right")
            )
            action = np.zeros([12])
            action[7:10] = target_base_pose_to_action(next_pose, curr_pose,
                                                      prev_pose, dt,
                                                      legacy=legacy)
            prev_pose = curr_pose.copy()
            obs, _, _, _ = env.step(action)
            unwrapped_env.sim.data.qpos[
                unwrapped_env.robots[0]._ref_joint_pos_indexes
            ] = reset_joint_pos
            unwrapped_env.sim.forward()

            if step_callback is not None:
                step_callback()

            if not legacy:
                # detect if robot is stuck and skip to next waypoint if so
                base_velp = unwrapped_env.sim.data.get_body_xvelp(robot_base_name)
                vel_norm = np.linalg.norm(base_velp[:2])
                cmd_mag = np.linalg.norm(action[7:9])
                if verbose:
                    print(f"command mag: {cmd_mag}; vel norm: {vel_norm:.4f}; vel/cmd {vel_norm / cmd_mag:.4f}")

                if vel_norm / (cmd_mag + 1e-12) < 0.02:
                    stuck_counter += 1
                else:
                    stuck_counter = 0

                if stuck_counter >= 5 and i < len(waypoints) - 2:
                    if verbose:
                        print(f"[Waypoint {i+1}] stuck for {stuck_counter} steps → skipping ahead")
                    break

            # Update base history
            base_pos, base_rot = robot_controller.get_controller_base_pose("right")
            base_heading = T.quat2axisangle(T.mat2quat(base_rot))[2]
            base_vec = np.array([base_pos[0], base_pos[1], base_heading])
            base_pos_history.append(base_pos.copy())
            base_mat_history.append(base_rot.copy())
            base_vec_history.append(base_vec)

            error = np.abs(base_vec - target_vec)
            step_error = np.abs(base_vec - next_vec)
            base_error_history.append(error)

            base_velp = unwrapped_env.sim.data.get_body_xvelp(robot_base_name)
            base_velr = unwrapped_env.sim.data.get_body_xvelr(robot_base_name)
            base_cmd = action[7:9]
            base_velp_history.append(base_velp.copy())
            base_velr_history.append(base_velr.copy())
            base_cmd_history.append(base_cmd.copy())

            if save_obj_data:
                bbox, pose_median, obj_detect = get_object_data(unwrapped_env)
                bbox_history.append(bbox)
                pose_median_history.append(pose_median)
                obj_detect_history.append(obj_detect)

            if verbose:
                curr_vec = np.array([base_pos[0], base_pos[1], base_heading])
                print(
                    f"[Waypoint {i + 1:02d} Step {t + 1:02d} / {num_steps:02d}] "
                    + f"current {curr_vec.round(3)}; target {target_vec.round(3)}; "
                    f"error {error.round(3)}; step error {step_error.round(3)}"
                )

            if render:
                if "fisheye" in camera_name:
                    img = unwrapped_env.sim.render(width=1024, height=1024,
                                                   camera_name=camera_name)[::-1]
                else:
                    img = obs[f"{camera_name}_image"][-1].transpose(1, 2, 0)
                image_history.append((img.copy() * 255).astype(np.uint8))

            if save_obj_data:
                bbox, pose_median, obj_detect = get_object_data(unwrapped_env)
                bbox_history.append(bbox)
                pose_median_history.append(pose_median)
                obj_detect_history.append(obj_detect)

            if i < len(waypoints) - 2 and (
                np.all(step_error < 0.005)
                or (t > 10 and np.all(np.abs(error - base_error_history[-11]) < 0.001))
            ):
                break
            if i == len(waypoints) - 2 and np.all(step_error < (0.001 if legacy else 0.002)):
                break

    info = dict(
        base_pos_history=base_pos_history,
        base_mat_history=base_mat_history,
        base_vec_history=base_vec_history,
        base_velp_history=base_velp_history,
        base_velr_history=base_velr_history,
        base_cmd_history=base_cmd_history,
        error_history=base_error_history,
        images=image_history,
    )
    if save_obj_data:
        info.update(dict(
            bbox_history=bbox_history,
            pose_median_history=pose_median_history,
            obj_detect_history=obj_detect_history
        ))

    print(
        f"[nav_utils.py] Navigated to {target_vec.round(3)} with error {error.round(3)}"
    )

    return info
