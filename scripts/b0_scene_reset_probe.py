"""One reset-only, outcome-blind B0 compiler integration probe."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from mobiwam.b0_scene_compiler import fixture_record, validate_fixture, validate_native_frame


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=("CloseSingleDoor", "CloseDrawer"), required=True)
    p.add_argument("--layout", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    from PIL import Image
    from robomimic.config import config_factory
    from robomimic.utils.file_utils import maybe_dict_from_checkpoint
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    config_path = a.checkpoint.parent.parent / "config.json"
    config_data = json.loads(config_path.read_text())
    config = config_factory(config_data["algo_name"])
    with config.values_unlocked():
        config.update(config_data)
    ObsUtils.initialize_obs_utils_with_config(config)
    checkpoint = maybe_dict_from_checkpoint(ckpt_path=str(a.checkpoint))
    meta = copy.deepcopy(checkpoint["env_metadata"])
    cameras = list(meta["env_kwargs"].get("camera_names", []))
    if "freeview" not in cameras:
        cameras.append("freeview")
    meta["env_kwargs"].update({"layout_and_style_ids": [[a.layout, a.layout]], "seed": a.seed, "camera_names": cameras})
    env = EnvUtils.create_env_from_metadata(env_meta=meta, env_name=meta["env_name"], render=False, render_offscreen=True, use_image_obs=True)
    # reset only: no env.step(), no task checker, no policy object.
    env.reset()
    raw = getattr(env, "env", env)
    fixture = raw.door_fxtr if a.task == "CloseSingleDoor" else raw.drawer
    fixture_name = next(name for name, value in raw.fixtures.items() if value is fixture)
    record = fixture_record(fixture_name, fixture, raw.sim)
    validate_fixture(a.task, record)
    frame = np.asarray(raw.sim.render(camera_name="freeview", width=1920, height=1080))[::-1]
    validate_native_frame(frame)
    image_path = a.output / "reset-freeview-1920x1080.png"
    Image.fromarray(frame).save(image_path)
    result = {"task": a.task, "layout": a.layout, "style": a.layout, "seed": a.seed, "reset_count": 1, "route_outcome_read": False, "env_step_calls": 0, "fixture": record, "camera": {"name": "freeview", "native_shape": list(frame.shape), "frame_path": str(image_path), "frame_sha256": digest(image_path)}, "checkpoint": {"path": str(a.checkpoint), "sha256": digest(a.checkpoint)}, "status": "pass"}
    (a.output / "probe.json").write_text(json.dumps(result, indent=2) + "\n")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
