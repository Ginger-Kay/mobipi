"""
Utilities related to loading policy checkpoints.

@yjy0625
"""
import copy
import json
import os
import numpy as np
from glob import glob

import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.lang_utils as LangUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.config import config_factory

def replace_dataset_path(path, data_root_dir):
    d1 = data_root_dir
    d2 = "/".join(path.split("/")[path.split("/").index("single_stage") - 1:])
    return os.path.join(d1, d2)


def get_config_for_policy(ckpt_root_dir, data_root_dir, env_name, method_name, seed=1, dataset_name="mg-300"):
    search_regex = os.path.join(ckpt_root_dir, "robocasa", method_name, "*-" + env_name,
                                f"seed_{seed}_*_{dataset_name}",
                                "*/models/model_epoch_*.pth")
    print(f"Searching for policy in regex [{search_regex}]")
    valid_ckpt_paths = glob(search_regex)
    folder_timestamps = [int(s.split("/")[-3]) if s.split("/")[-3].isdigit() else -1 for s in valid_ckpt_paths]
    most_recent_timestamp = np.max(folder_timestamps)
    filtered_ckpt_paths = [path for i, path in enumerate(valid_ckpt_paths) \
                           if folder_timestamps[i] == most_recent_timestamp]
    ckpt_idxs = [int(s.split("/")[-1].split(".")[0].split("_")[-1]) for s in filtered_ckpt_paths]
    ckpt_path = filtered_ckpt_paths[np.argmax(ckpt_idxs)]
    config_path = os.path.join(os.path.dirname(ckpt_path), "../config.json")
    print(f"Loading config at {config_path}")
    ext_cfg = json.load(open(config_path, 'r'))
    config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)
        config["runtime_checkpoint_path"] = ckpt_path
        config["train"]["data"][0]["path"] = replace_dataset_path(
            config["train"]["data"][0]["path"], data_root_dir
        )
    return config, ckpt_path


def load_policy(config, ckpt_path):
    ObsUtils.initialize_obs_utils_with_config(config)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    checkpoint = FileUtils.maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
    shape_meta = checkpoint["shape_metadata"]

    lang_encoder = LangUtils.LangEncoder(
        device=device,
    )
    obs_normalization_stats = copy.deepcopy(checkpoint.get("obs_normalization_stats", None))
    action_normalization_stats = copy.deepcopy(checkpoint.get("action_normalization_stats", None))
    for stats in (obs_normalization_stats, action_normalization_stats):
        if stats is not None:
            for modality in stats.values():
                for key, value in modality.items():
                    modality[key] = np.array(value)

    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    print("LOADING MODEL WEIGHTS FROM " + ckpt_path)
    model.deserialize(checkpoint["model"])
    model.set_eval()

    rollout_model = RolloutPolicy(
        model,
        obs_normalization_stats=obs_normalization_stats,
        action_normalization_stats=action_normalization_stats,
        lang_encoder=lang_encoder,
    )

    return model, rollout_model, lang_encoder
