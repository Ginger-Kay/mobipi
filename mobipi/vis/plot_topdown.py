"""
Standalone script to render a topdown image of a sim scene for visualization.

Usage:
Fill in the ckpt_path variable below the imports and run `python plot_topdown.py`

@yjy0625
"""
import os
import json
import numpy as np
import torch
import pickle
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors
import matplotlib.patches as patches

from robosuite.utils.transform_utils import mat2quat
from robosuite.utils.mjmod import CameraModder

from robomimic.utils.script_utils import deep_update
from robomimic.config import config_factory
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils

ckpt_path = "[enter path to a trained policy checkpoint (*.pth)]"
if not ckpt_path.endswith(".pth"):
    print("Please insert [ckpt_path] in the code to render a scene.")
    exit(0)
config_path = os.path.join(os.path.dirname(ckpt_path), "../config.json")
ext_cfg = json.load(open(config_path, 'r'))
config = config_factory(ext_cfg["algo_name"])

# update config with external json - this will throw errors if
# the external config has keys not present in the base algo config
with config.values_unlocked():
    config.update(ext_cfg)
    config.train.hdf5_use_swmr = False

# first set seeds
np.random.seed(config.train.seed)
torch.manual_seed(config.train.seed)

# set num workers
torch.set_num_threads(1)

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

for (dataset_i, dataset_cfg) in enumerate(config.train.data):
    do_eval = dataset_cfg.get("do_eval", True)
    if do_eval is not True:
        continue
    env_meta = env_meta_list[dataset_i]
    shape_meta = shape_meta_list[dataset_i]
    env_name = env_meta_list[dataset_i]["env_name"]
    horizon = dataset_cfg.get("horizon", config.experiment.rollout.horizon)
    break

env_meta["env_kwargs"]["layout_ids"] = None
env_meta["env_kwargs"]["style_ids"] = None
env_meta["env_kwargs"]["layout_and_style_ids"] = [[0, 1]]
env_meta["env_kwargs"]["place_robot"] = False
    
env_kwargs = dict(
    env_meta=env_meta,
    env_name=env_name,
    render=False,
    render_offscreen=config.experiment.render_video,
    use_image_obs=shape_meta["use_images"],
    seed=0,
)

env = EnvUtils.create_env_from_metadata(**env_kwargs)
env = EnvUtils.wrap_env_from_config(env, config=config)

# hide robot
unwrapped_env = env.unwrapped.env
for i in range(unwrapped_env.sim.model.ngeom):
    geom_name = unwrapped_env.sim.model.geom_id2name(i)
    if "robot0" in geom_name or "base0" in geom_name or "gripper0" in geom_name:
        unwrapped_env.sim.model.geom_rgba[i, 3] = 0  # Set alpha to 0 for invisibility

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


camera_name = "robot0_robotview"
xy_center = (0.8, -1.5)  # Center of the scene
z_height = 450
rotation_angle = 180
image_size = (1024, 1024)  # Image resolution
xy_range = (2.0, 2.0)  # Width and height of the orthographic view in world units

camera_params = setup_orthographic_camera(
    env.unwrapped.env, camera_name, xy_center, z_height, rotation_angle / 180 * np.pi, fovy=0.3
)
image = env.unwrapped.env.sim.render(width=image_size[0], height=image_size[1], camera_name=camera_name)[::-1]

# compute 2d locations of image corners
image_corners = unproject_image_corners_on_floor(camera_params["pos"], camera_params["quat"], camera_params["fovy"], image_size)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image, extent=(image_corners[0, 0], image_corners[1, 0], image_corners[2, 1], image_corners[0, 1]),
         zorder=1)

data_dir = "eval_policy_1221"

with open(os.path.join(data_dir, "results.pkl"), "rb") as f:
    results = pickle.load(f)
penetration_filename = os.path.join(data_dir, "penetrations.pkl")
if os.path.isfile(penetration_filename):
    with open(penetration_filename, "rb") as f:
        penetrations = pickle.load(f)["penetration"]
else:
    penetrations = [0] * len(results)

x_min, x_max, x_step = 0.75, 1.75, 21
y_min, y_max, y_step = -1.7, -0.7, 21
x_values = np.linspace(x_min, x_max, x_step)
y_values = np.linspace(y_min, y_max, y_step)
step_size_x = (x_max - x_min) / (x_step - 1)
step_size_y = (y_max - y_min) / (y_step - 1)
xv, yv = np.meshgrid(x_values, y_values)

reward_grid = np.array(results).reshape(len(y_values), len(x_values))
penetrations_grid = np.array(penetrations).reshape(len(y_values), len(x_values))
vis_grid = reward_grid.copy()
vis_grid[penetrations_grid > 0] = -1

# Define the custom colormap
def custom_colormap(value):
    if value < 0:
        return [0.5, 0.5, 0.5, 1]  # Gray color (RGBA)
    else:
        return plt.cm.viridis(value)  # Viridis colormap for 0 to 1

# Create a custom normalization and colormap for the values
class CustomNormalize(Normalize):
    def __call__(self, value, clip=None):
        return super().__call__(np.clip(value, 0, 1), clip)

# Map values to the custom colormap
mapped_cmap = ListedColormap([custom_colormap(value) for value in np.linspace(-1, 1, x_step)])
norm = mcolors.Normalize(vmin=-1, vmax=1)

for i in range(xv.shape[0]):
    for j in range(xv.shape[1]):
        x, y, vis_value = xv[i, j], yv[i, j], vis_grid[i, j]
        rect = patches.Rectangle((x, y), step_size_x, step_size_y, linewidth=0, edgecolor='none', facecolor=mapped_cmap(norm(vis_value)), alpha=0.8)
        plt.gca().add_patch(rect)

# Plot ID robot position
ax.scatter(1.28, -0.8, s=20, marker='*', color='red', zorder=3)

# Plot nav path
nav_data = np.load("eval_policy/faucet_wideang_sample_nav_5.npz")
x, y = nav_data["base_pos"][:, 0], nav_data["base_pos"][:, 1]
ax.plot(x, y, marker='o', linestyle='-', color='gray')

plt.savefig("plot_topdown_faucet5_wide.pdf")
