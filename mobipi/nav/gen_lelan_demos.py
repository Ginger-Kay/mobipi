"""
Script for generating in-distribution navigation data for LeLaN fine-tuning.

Usage:
    python gen_lelan_demos.py --env_name CloseSingleDoor

@yjy0625
"""
import os
import torch
import click

import cv2
import math
import pickle
from glob import glob
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import robosuite.utils.transform_utils as T
from robosuite.utils.camera_utils import get_real_depth_map
from robocasa.utils.env_utils import create_env, set_robot_base_pose
import mimicgen.utils.pose_utils as PoseUtils

from mobipi.macros import *
from mobipi.utils.constants import *
from mobipi.utils.nav_utils import move_to_pose
from mobipi.utils.media_utils import add_caption_to_img, to_fisheye


class GTCollisionChecker:
    def __init__(self, env, robot_size=0, vis=False):
        self.env = env
        self.robot_size = robot_size # unused
        self.vis = vis
        self.vis_images = []

    def _check_collision(self):
        sim = self.env.sim
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]

            # Get names of the two geoms in contact
            geom1 = sim.model.geom_id2name(contact.geom1)
            geom2 = sim.model.geom_id2name(contact.geom2)

            # Skip if names not found (safety check)
            if geom1 is None or geom2 is None:
                continue

            # Check if either geom belongs to the robot or mobile base
            is_robot1 = geom1.startswith("robot0") or geom1.startswith("mobilebase0")
            is_robot2 = geom2.startswith("robot0") or geom2.startswith("mobilebase0")

            # If one is robot and the other is not from robot/mobile base, it's a collision
            if is_robot1 != is_robot2:
                return True

        return False

    def compute_score(self, _unused0, _unused1, pose_arr, numpy=True, robot_size=0):
        assert numpy
        scores = []
        for pose in pose_arr:
            current_state = self.env.sim.get_state()
            set_robot_base_pose(self.env, pose)
            col_free = not self._check_collision()
            if self.vis:
                img = self.env.sim.render(camera_name="robot0_frontview", width=512, height=512)[::-1]
                info_dict = {
                    "pose": ", ".join([f"{p:.3f}" for p in pose]),
                    "col_free": "true" if col_free else "false"
                }
                vis_img = add_caption_to_img(img, info_dict)
                self.vis_images.append(vis_img)
            self.env.sim.set_state(current_state)
            self.env.sim.forward()
            # print(f"Pose {pose} is collision free? {col_free}")
            scores.append(col_free)
        return np.array(scores)


def compute_non_policy_aware_base_pose(env, rng, fov=np.pi * 75 / 180):    
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

    offset = np.array([0., 0.])
    if hasattr(env, "drawer"):
        offset[1] = -0.23
    offset[0] += rng.uniform(-0.5, 0.5)
    offset[1] -= rng.uniform(0, 0.5)
    print(offset)
    pos, ori = env.compute_robot_base_placement_pose(ref)
    print(pos, ori)
    pos, ori = env.compute_robot_base_placement_pose(ref, offset=tuple(offset))
    print(pos, ori)
    ori[-1] += rng.uniform(-fov / 2, fov / 2)
    return pos, ori


@click.command()
@click.option('--env_name', default='TurnOnSinkFaucet', type=str,
              help='Environment name.')
@click.option('--seed', default=1, type=int, help='Random seed.')
@click.option('--layout_ids',
              default=",".join([str(i) for i in TRAIN_LAYOUTS]), type=str,
              help='Layout IDs.')
@click.option('--num_demos', default=110, type=int, help='Number of demonstrations.')
@click.option('--data_root_dir', default=os.path.join(DATA_ROOT_DIR, "lelan_ft"), type=str,
              help='Root directory of the dataset.')
@click.option('--skip_existing', is_flag=True)
@click.option('--overwrite', is_flag=True)
@click.option('--policy_aware', is_flag=True)
def main(env_name, seed, layout_ids, num_demos, data_root_dir, skip_existing, overwrite, policy_aware):
    layout_ids = [int(s) for s in layout_ids.split(",")]
    pbar = tqdm(total=len(layout_ids) * num_demos, desc="Collecting lelan demos")
    for layout_id in layout_ids:
        env = create_env(env_name=env_name, seed=seed, layout_ids=layout_id, style_ids=layout_id, place_robot=True,
                         obj_instance_split="A", place_robot_for_nav=True, generative_textures="100p")

        num_collected_demos = 0
        trial_idx = 0
        while num_collected_demos < num_demos:
            env.reset()
            if policy_aware:
                default_init_pos = env._default_init_robot_pos
                default_init_ori = env._default_init_robot_ori
            else:
                rng = np.random.default_rng(seed=trial_idx * 100 + layout_id)
                default_init_pos, default_init_ori = compute_non_policy_aware_base_pose(env, rng)
            collision_checker = GTCollisionChecker(env)

            obj_name = env.get_object_name()

            nav_info = move_to_pose(
                env,
                PoseUtils.make_pose(default_init_pos, T.quat2mat(T.axisangle2quat(default_init_ori))),
                render=True,
                camera_name="robot0_fisheye",
                unwrapped=True,
                verbose=False,
                k_col=collision_checker,
                save_obj_data=True
            )

            final_error = nav_info["error_history"][-1]
            final_error[-1] = np.mod(final_error[-1] + np.pi, np.pi * 2) - np.pi
            if np.all(final_error < 0.01) and np.all(nav_info["obj_detect_history"]):
                formatted_filename = f"{env_name}_layout{layout_id}_seed{seed}_ep{num_collected_demos:04d}"
                save_dir = os.path.join(data_root_dir + ("" if policy_aware else "_npa"),
                                        formatted_filename)
                if os.path.exists(save_dir):
                    if skip_existing:
                        num_collected_demos += 1
                        pbar.update(1)
                        continue
                    if overwrite:
                        ans = "y"
                    else:
                        ans = input(f"Directory {save_dir} exists. Overwrite? Answer 'y' or 'n'")
                    if ans == "y":
                        os.system(f"rm -rf {save_dir}")
                    else:
                        ans = input(f"Skip all later existing directories? Answer 'y' or 'n'")
                        if ans == "y":
                            skip_existing = True
                        num_collected_demos += 1
                        pbar.update(1)
                        continue
                image_dir = os.path.join(save_dir, "image")
                pickle_dir = os.path.join(save_dir, "pickle")
                os.makedirs(image_dir)
                os.makedirs(pickle_dir)

                # save episode
                for t in range(len(nav_info["images"])):
                    image = 255 - nav_info["images"][t]
                    fisheye_image = to_fisheye(image)
                    cv2.imwrite(os.path.join(image_dir, f"{t:08d}.jpg"),
                                cv2.resize(fisheye_image, (224, 224))[..., ::-1])
                    step_info = {
                        "bbox": nav_info["bbox_history"][t],
                        "pose_median": nav_info["pose_median_history"][t],
                        "obj_detect": nav_info["obj_detect_history"][t]
                    }
                    if step_info["obj_detect"]:
                        step_info["prompt"] = [obj_name]
                    with open(os.path.join(pickle_dir, f"{t:08d}.pkl"), "wb") as f:
                        pickle.dump(step_info, f)

                num_collected_demos += 1
                pbar.update(1)
                print(f"Collected demo {formatted_filename}")
            else:
                print(f"Demo collection failed")

            del nav_info
            trial_idx += 1


if __name__ == "__main__":
    main()
