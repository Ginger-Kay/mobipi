"""Safely validate and atomically install the frozen B0 official scene assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


TASKS = ("close_drawer", "close_single_door")
CELLS = (1, 4, 7, 8, 9)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_archives() -> tuple[str, ...]:
    return tuple(f"{task}_layout{i}_style{i}.zip" for task in TASKS for i in CELLS)


def safe_members(archive: zipfile.ZipFile, expected_top: str, *, test_crc: bool = True) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    if not members:
        raise ValueError("empty archive")
    for info in members:
        name = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"unsafe archive path: {info.filename}")
        if len(name.parts) < 2 or "/".join(name.parts[:2]) != expected_top:
            raise ValueError(f"unexpected archive top level: {info.filename}")
        if stat.S_ISLNK(mode):
            raise ValueError(f"symlink forbidden: {info.filename}")
    if test_crc:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"CRC failure: {bad}")
    return members


def scene_inventory(scene: Path, task: str, cell: int) -> dict[str, object]:
    transforms = scene / "transforms.json"
    point_cloud = scene / "pc.ply"
    configs = list(scene.glob("model/splatfacto/*/config.yml"))
    parser_meta = list(scene.glob("model/splatfacto/*/dataparser_transforms.json"))
    checkpoints = list(scene.glob("model/splatfacto/*/nerfstudio_models/*.ckpt"))
    if not (transforms.is_file() and point_cloud.is_file() and len(configs) == len(parser_meta) == len(checkpoints) == 1):
        raise ValueError(f"incomplete 3DGS scene: {scene}")
    transform_data = json.loads(transforms.read_text())
    parser_data = json.loads(parser_meta[0].read_text())
    with point_cloud.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"invalid PLY header: {point_cloud}")
    if "transform" not in parser_data or "scale" not in parser_data:
        raise ValueError(f"incomplete dataparser metadata: {parser_meta[0]}")
    files = [p for p in scene.rglob("*") if p.is_file()]
    return {
        "task": task,
        "layout": cell,
        "style": cell,
        "path": str(scene),
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "image_frames": len(list((scene / "images").glob("*.png"))),
        "metadata_frames": len(transform_data.get("frames", [])),
        "point_cloud": {"path": str(point_cloud), "bytes": point_cloud.stat().st_size, "sha256": sha256(point_cloud)},
        "config": {"path": str(configs[0]), "bytes": configs[0].stat().st_size, "sha256": sha256(configs[0])},
        "dataparser": {"path": str(parser_meta[0]), "bytes": parser_meta[0].stat().st_size, "sha256": sha256(parser_meta[0])},
        "checkpoint": {"path": str(checkpoints[0]), "bytes": checkpoints[0].stat().st_size, "sha256": sha256(checkpoints[0])},
        "loader_smoke": "metadata_and_binary_structure_pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--preverified-exit-code", type=Path)
    args = parser.parse_args()
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    archives = expected_archives()
    sums: dict[str, str] = {}
    for line in (args.download_root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.lstrip("* ")] = digest
    if set(archives) != set(sums):
        raise ValueError("SHA256SUMS does not exactly bind the ten frozen archives")
    preverified = args.preverified_exit_code is not None
    if preverified and args.preverified_exit_code.read_text().strip() != "0":
        raise ValueError("external SHA/CRC verification did not pass")
    if sum((args.download_root / name).stat().st_size for name in archives) != 9_242_655_772:
        raise ValueError("unexpected aggregate ZIP bytes")
    if args.runtime_root.exists():
        raise FileExistsError(f"runtime root already exists: {args.runtime_root}")
    args.staging_root.mkdir(parents=True, exist_ok=False)
    archive_rows = []
    for name in archives:
        path = args.download_root / name
        actual = sums[name] if preverified else sha256(path)
        if not preverified and actual != sums[name]:
            raise ValueError(f"SHA mismatch: {name}")
        task = name.rsplit("_layout", 1)[0]
        cell = int(name.rsplit("_layout", 1)[1].split("_", 1)[0])
        expected_top = f"{task}/layout{cell}_style{cell}"
        with zipfile.ZipFile(path) as archive:
            members = safe_members(archive, expected_top, test_crc=not preverified)
            archive.extractall(args.staging_root)
        archive_rows.append({"name": name, "bytes": path.stat().st_size, "sha256": actual, "members": len(members), "crc": "pass"})
    scenes = [scene_inventory(args.staging_root / task / f"layout{i}_style{i}", task, i) for task in TASKS for i in CELLS]
    all_files = [p for p in args.staging_root.rglob("*") if p.is_file()]
    inventory = {
        "schema_version": "b0-official-assets-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hf_repo": "Ginger-Kay/mobipi-scene-model-zips",
        "hf_revision": "14efb229b1e046c7f588401be3048a806f298dae",
        "download_root": str(args.download_root.resolve()),
        "runtime_root": str(args.runtime_root.resolve()),
        "upstream_checksum": "upstream_checksum_unavailable",
        "archives": archive_rows,
        "scenes": scenes,
        "total_files": len(all_files),
        "total_bytes": sum(p.stat().st_size for p in all_files),
        "status": "pass",
    }
    manifest = args.artifact_root / "asset-inventory.json"
    manifest.write_text(json.dumps(inventory, indent=2) + "\n")
    os.rename(args.staging_root, args.runtime_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
