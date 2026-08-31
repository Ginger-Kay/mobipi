#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


BASE_URL = "https://download.cs.stanford.edu/juno/mobipi/pi"
PROJECT_ROOT = Path("/share/jhk/MobiWAM")
ARCHIVES = {
    "CloseDrawer": "04-12-CloseDrawer_seed_1_CloseDrawer_mg-300.zip",
    "TurnOnFaucet": "04-12-TurnOnFaucet_seed_1_TurnOnSinkFaucet_mg-300.zip",
    "TurnOnMicrowave": "04-12-TurnOnMicrowave_seed_1_TurnOnMicrowave_mg-300.zip",
    "TurnOnStove": "04-12-TurnOnStove_seed_1_TurnOnStove_mg-300.zip",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_with_resume(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "MMWAM-OBC-001/1.0"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        status = getattr(response, "status", response.getcode())
        if offset and status != 206:
            raise RuntimeError("server ignored Range resume; refusing to overwrite partial")
        mode = "ab" if offset else "wb"
        with partial.open(mode) as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                output.flush()
    os.replace(partial, destination)


def validate_members(archive: zipfile.ZipFile) -> None:
    for info in archive.infolist():
        member = PurePosixPath(info.filename)
        if member.is_absolute() or ".." in member.parts:
            raise RuntimeError(f"unsafe ZIP path: {info.filename}")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise RuntimeError(f"ZIP symlink is forbidden: {info.filename}")


def extract_checkpoint(task: str, archive_path: Path, destination_root: Path) -> dict[str, object]:
    expected_top = archive_path.stem.split("_seed", 1)[0]
    final_top = destination_root / expected_top
    if final_top.exists():
        raise FileExistsError(f"refusing to merge checkpoint tree: {final_top}")
    staging = PROJECT_ROOT / "cache" / "downloads" / "mobipi-tasks" / "staging" / archive_path.stem
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    staging.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        validate_members(archive)
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad_member}")
        archive.extractall(staging)
    staged_top = staging / expected_top
    models = sorted(staged_top.glob("**/models/model_epoch_*.pth"))
    if len(models) != 1:
        raise RuntimeError(f"expected exactly one checkpoint, observed {models}")
    destination_root.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staged_top), str(final_top))
    staging.rmdir()
    model = final_top / models[0].relative_to(staged_top)
    return {
        "task": task,
        "url": f"{BASE_URL}/{archive_path.name}",
        "archive": str(archive_path),
        "archive_size": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "zip_crc": "pass",
        "checkpoint": str(model),
        "checkpoint_size": model.stat().st_size,
        "checkpoint_sha256": sha256_file(model),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download exact missing Mobi-pi task checkpoints")
    parser.add_argument("--tasks", nargs="+", choices=sorted(ARCHIVES), required=True)
    parser.add_argument(
        "--record",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "MMWAM-OBC-001" / "setup" / "task-checkpoints.json",
    )
    args = parser.parse_args()
    if args.record.exists():
        raise FileExistsError(f"refusing to overwrite download record: {args.record}")
    download_root = PROJECT_ROOT / "cache" / "downloads" / "mobipi-tasks"
    destination_root = PROJECT_ROOT / "checkpoints" / "MMWAM-OBC-001" / "robocasa" / "bc_xfmr"
    records = []
    for task in args.tasks:
        filename = ARCHIVES[task]
        archive_path = download_root / filename
        if not archive_path.exists():
            download_with_resume(f"{BASE_URL}/{filename}", archive_path)
        records.append(extract_checkpoint(task, archive_path, destination_root))
    payload = {
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "official Mobi-pi download server",
        "artifacts": records,
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.record)


if __name__ == "__main__":
    main()
