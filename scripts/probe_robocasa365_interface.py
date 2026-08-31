#!/usr/bin/env python3
"""Create one RoboCasa365 task and record the port-facing interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import robocasa  # noqa: F401 - importing registers Gymnasium environments


DEFAULT_ROOT = Path("/share/jhk/MobiWAM")


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_ROOT / "configs" / "robocasa365_port.json"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_ROOT / "audit" / "robocasa365_interface.json"
    )
    args = parser.parse_args()

    root = args.root.resolve()
    allowed_root = Path("/share/chensiyu").resolve()
    if allowed_root not in root.parents:
        raise RuntimeError(f"Project root must stay under {allowed_root}: {root}")
    config = json.loads(args.config.read_text(encoding="utf-8"))

    env = gym.make(
        config["environment_id"],
        split=config["split"],
        seed=int(config["seed"]),
    )
    try:
        reset_result = env.reset()
        observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        raw = env.unwrapped
        robots = getattr(raw, "robots", [])
        robot = robots[0] if robots else None
        report = {
            "environment_id": config["environment_id"],
            "split": config["split"],
            "seed": config["seed"],
            "action_space": repr(env.action_space),
            "action_shape": list(env.action_space.shape),
            "observation_space": repr(env.observation_space),
            "observation": {
                str(key): jsonable(value) for key, value in observation.items()
            },
            "raw_environment_class": type(raw).__name__,
            "robot_class": type(robot).__name__ if robot is not None else None,
            "robot_model_class": (
                type(robot.robot_model).__name__ if robot is not None else None
            ),
            "robot_arms": list(getattr(robot, "arms", [])) if robot is not None else [],
            "eef_site_id": repr(getattr(robot, "eef_site_id", None)),
            "controller_class": (
                type(getattr(robot, "composite_controller", None)).__name__
                if robot is not None
                else None
            ),
            "has_get_state": callable(getattr(raw, "get_state", None)),
            "has_reset_to": callable(getattr(raw, "reset_to", None)),
        }
    finally:
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
