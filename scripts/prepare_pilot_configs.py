#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path("/share/jhk/MobiWAM")
REPO = PROJECT_ROOT / "Mobipi"
CLOSE_DOOR_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints/inherited/chensiyu-20260830/robocasa/bc_xfmr/04-12-CloseSingleDoor/"
    "seed_1_CloseSingleDoor_mg-300/20250413055045/models/model_epoch_1000.pth"
)
CLOSE_DOOR_SHA256 = "6cafee55eaf087a93b6e604d072da459c6200b15616f14c32120e29f32be9852"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze four task-specific paired-pilot configs")
    parser.add_argument("--asset-record", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite pilot config root: {args.output_root}")
    status = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True)
    if status:
        raise RuntimeError("pilot config freeze requires a clean code tree")
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    base = json.loads((REPO / "configs/mobipi_close_single_door.json").read_text(encoding="utf-8"))
    record = json.loads(args.asset_record.read_text(encoding="utf-8"))
    task_assets = {item["task"]: item for item in record["artifacts"]}
    task_assets["CloseSingleDoor"] = {
        "checkpoint": str(CLOSE_DOOR_CHECKPOINT),
        "checkpoint_sha256": CLOSE_DOOR_SHA256,
    }
    args.output_root.mkdir(parents=True)
    manifest = {"code_commit": commit, "configs": {}}
    for task in ("CloseSingleDoor", "CloseDrawer", "TurnOnFaucet", "TurnOnMicrowave"):
        payload = dict(base)
        payload.update(
            {
                "env_name": task,
                "code_commit": commit,
                "checkpoint_root": (
                    str(PROJECT_ROOT / "checkpoints/inherited/chensiyu-20260830")
                    if task == "CloseSingleDoor"
                    else str(PROJECT_ROOT / "checkpoints/MMWAM-OBC-001")
                ),
                "policy_checkpoint_path": task_assets[task]["checkpoint"],
                "policy_checkpoint_hash": task_assets[task]["checkpoint_sha256"],
                "layouts": [1, 4, 7, 8, 9],
                "base_noise_sigmas": [0.0, 0.10, 0.20],
                "states_per_noise_per_layout": 1,
                "save_video": False,
            }
        )
        content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        path = args.output_root / f"{task}.json"
        path.write_bytes(content)
        manifest["configs"][task] = {"path": str(path), "sha256": sha256_bytes(content)}
    manifest_content = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (args.output_root / "manifest.json").write_bytes(manifest_content)
    print(args.output_root / "manifest.json")


if __name__ == "__main__":
    main()
