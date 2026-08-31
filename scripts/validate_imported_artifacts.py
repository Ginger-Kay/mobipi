#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import torch


PROJECT_ROOT = Path("/share/jhk/MobiWAM")
CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints/inherited/chensiyu-20260830/robocasa/bc_xfmr/04-12-CloseSingleDoor/"
    "seed_1_CloseSingleDoor_mg-300/20250413055045/models/model_epoch_1000.pth"
)
DATASET = (
    PROJECT_ROOT
    / "data/inherited/chensiyu-20260830/v0.1/single_stage/kitchen_doors/"
    "CloseSingleDoor/mg/2024-05-04-22-34-56/demo_im128_fixview.hdf5"
)
CLIP_WEIGHT = (
    PROJECT_ROOT
    / "cache/huggingface/hub/models--openai--clip-vit-large-patch14/blobs/"
    "f1a17cdbe0f36fec524f5cafb1c261ea3bbbc13e346e0f74fc9eb0460dedd0d3"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def torch_record(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected dict checkpoint: {path}")
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "format": "torch-dict",
        "top_level_keys": sorted(str(key) for key in payload)[:100],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate copied checkpoint/data/encoder formats")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/MMWAM-OBC-001/setup/imported-artifact-integrity.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite integrity record: {args.output}")
    checkpoint = torch_record(CHECKPOINT)
    clip = torch_record(CLIP_WEIGHT)
    with h5py.File(DATASET, "r") as handle:
        dataset = {
            "path": str(DATASET.resolve()),
            "size": DATASET.stat().st_size,
            "format": "HDF5",
            "top_level_keys": sorted(handle.keys()),
            "demo_count": len(handle["data"]),
            "filter_300_demos_count": len(handle["mask"]["300_demos"]),
        }
    record = {
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "pass",
        "checkpoint": checkpoint,
        "dataset": dataset,
        "frozen_encoder": clip,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
