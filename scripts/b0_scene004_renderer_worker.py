#!/usr/bin/env python3
"""Fresh executable SCENE-004 renderer worker.

Only standard-library modules are imported before EGL/GPU variables are set.
The worker restores frozen XML/state snapshots and never constructs a task
environment or calls reset, step, or an outcome checker.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device-id", type=int, required=True)
    args = parser.parse_args()
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.device_id)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.device_id))
    from mobiwam.scene004_renderer import render_seed_group, write_json

    payload = json.loads(args.payload.read_text())
    receipt = render_seed_group(payload, args.output_dir, args.device_id)
    receipt["payload_sha256"] = __import__("hashlib").sha256(args.payload.read_bytes()).hexdigest()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.receipt, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
