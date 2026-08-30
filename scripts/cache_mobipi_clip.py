#!/usr/bin/env python3
"""Cache and verify the CLIP model required by the frozen policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_ROOT = Path("/share/chensiyu/MobiWAM")
MODEL_ID = "openai/clip-vit-large-patch14"
MODEL_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
MODEL_WEIGHT = "pytorch_model.bin"
MODEL_WEIGHT_SIZE = 1_710_671_599
MODEL_WEIGHT_SHA256 = "f1a17cdbe0f36fec524f5cafb1c261ea3bbbc13e346e0f74fc9eb0460dedd0d3"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--revision",
        default=MODEL_REVISION,
        help="Pinned Hugging Face commit; the downloaded weight is also checksum-pinned",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    allowed_root = Path("/share/chensiyu").resolve()
    if allowed_root not in root.parents:
        raise RuntimeError(f"Project root must stay under {allowed_root}: {root}")

    hf_home = root / "data" / "cache" / "huggingface" / "hf-home"
    hub_cache = hf_home / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)

    snapshot_path = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=args.revision,
            cache_dir=hub_cache,
            allow_patterns=[
                "config.json",
                "merges.txt",
                "preprocessor_config.json",
                "pytorch_model.bin",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
            ],
            resume_download=True,
        )
    )

    weight = snapshot_path / MODEL_WEIGHT
    if not weight.is_file():
        raise FileNotFoundError(weight)
    if weight.stat().st_size != MODEL_WEIGHT_SIZE:
        raise RuntimeError(f"unexpected CLIP weight size: {weight.stat().st_size}")
    weight_hash = sha256_file(weight)
    if weight_hash != MODEL_WEIGHT_SHA256:
        raise RuntimeError(f"CLIP weight checksum mismatch: {weight_hash}")

    required_files = [
        "config.json",
        "merges.txt",
        "preprocessor_config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "vocab.json",
    ]
    missing = [name for name in required_files if not (snapshot_path / name).is_file()]
    if missing:
        raise RuntimeError(f"CLIP snapshot is incomplete: {missing}")

    record = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": MODEL_ID,
        "requested_revision": args.revision,
        "resolved_snapshot": snapshot_path.name,
        "snapshot_path": str(snapshot_path),
        "weight_path": str(weight),
        "weight_size": weight.stat().st_size,
        "weight_sha256": weight_hash,
        "hf_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
    }
    audit_path = root / "audit" / "mobipi_clip_cache.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"CLIP cache verified: {snapshot_path}")
    print(f"audit written: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
