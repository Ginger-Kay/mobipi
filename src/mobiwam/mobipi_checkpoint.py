from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def normalization_stats_to_numpy(
    stats: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, np.ndarray]] | None:
    if stats is None:
        return None
    return {
        str(action_key): {
            str(statistic): np.asarray(value)
            for statistic, value in values.items()
        }
        for action_key, values in stats.items()
    }


def load_policy_from_checkpoint(config: Any, checkpoint_path: Path | str) -> tuple[Any, ...]:
    """Load Mobi-pi without opening the 9.60 GB source HDF5.

    Robomimic checkpoints already carry the exact environment metadata, shape
    metadata, and action normalization statistics saved at training time.
    """

    import robomimic.utils.lang_utils as LangUtils
    import robomimic.utils.torch_utils as TorchUtils
    from robomimic.algo import RolloutPolicy, algo_factory
    from robomimic.utils.file_utils import maybe_dict_from_checkpoint

    checkpoint = maybe_dict_from_checkpoint(ckpt_path=str(checkpoint_path))
    required = {
        "model",
        "env_metadata",
        "shape_metadata",
        "action_normalization_stats",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise RuntimeError(f"checkpoint is missing required evaluation fields: {missing}")

    env_meta = copy.deepcopy(checkpoint["env_metadata"])
    shape_meta = copy.deepcopy(checkpoint["shape_metadata"])
    if int(shape_meta["ac_dim"]) != 12:
        raise RuntimeError(f"expected a 12-D checkpoint, got {shape_meta['ac_dim']}")
    action_stats = normalization_stats_to_numpy(
        checkpoint["action_normalization_stats"]
    )
    obs_stats = normalization_stats_to_numpy(
        checkpoint.get("obs_normalization_stats")
    )

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    language_encoder = LangUtils.LangEncoder(device=device)
    import robomimic.utils.obs_utils as ObsUtils
    ObsUtils.initialize_obs_utils_with_config(config)
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    model.deserialize(checkpoint["model"])
    rollout_policy = RolloutPolicy(
        model,
        obs_normalization_stats=obs_stats,
        action_normalization_stats=action_stats,
        lang_encoder=language_encoder,
    )
    return model, rollout_policy, language_encoder, env_meta, shape_meta


def create_env_from_checkpoint_metadata(
    config: Any,
    env_meta: Mapping[str, Any],
    shape_meta: Mapping[str, Any],
    override_args: Mapping[str, Any],
) -> Any:
    import torch
    import robomimic.utils.env_utils as EnvUtils

    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    metadata = copy.deepcopy(env_meta)
    metadata["env_kwargs"].update(dict(override_args))
    env = EnvUtils.create_env_from_metadata(
        env_meta=metadata,
        env_name=metadata["env_name"],
        render=False,
        render_offscreen=bool(shape_meta["use_images"])
        or bool(config.experiment.render_video),
        use_image_obs=shape_meta["use_images"],
    )
    return EnvUtils.wrap_env_from_config(env, config=config)
