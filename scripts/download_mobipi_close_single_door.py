#!/usr/bin/env python3
"""Download and verify only the first Mobi-pi task artifacts.

The official interactive downloader always couples a policy checkpoint with its
training dataset. This script keeps that behavior explicit, supports resume,
checks the published artifacts, and rejects unsafe ZIP members.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


DEFAULT_ROOT = Path("/share/chensiyu/MobiWAM")
BASE_URL = "https://download.cs.stanford.edu/juno/mobipi/pi"


@dataclass(frozen=True)
class Artifact:
    name: str
    filename: str
    sha256: str
    extract_root: str
    expected_relative_path: str
    expected_size: int

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.filename}"


CHECKPOINT = Artifact(
    name="CloseSingleDoor checkpoint seed 1",
    filename="04-12-CloseSingleDoor_seed_1_CloseSingleDoor_mg-300.zip",
    sha256="a1294905a059505c2c285d8f8fc3c13be396d4c3bbc0b91e3b3ead00c3fe3a0d",
    extract_root="checkpoints/robocasa/bc_xfmr",
    expected_relative_path=(
        "04-12-CloseSingleDoor/seed_1_CloseSingleDoor_mg-300/"
        "20250413055045/models/model_epoch_1000.pth"
    ),
    expected_size=246_511_905,
)

DATASET = Artifact(
    name="CloseSingleDoor mg dataset",
    filename="data_CloseSingleDoor.zip",
    sha256="b14a21a7d36b778b6b8203dbc404de6d40823ab741dd1e3bf5d635725e1bc99c",
    extract_root="data",
    expected_relative_path=(
        "v0.1/single_stage/kitchen_doors/CloseSingleDoor/mg/"
        "2024-05-04-22-34-56/demo_im128_fixview.hdf5"
    ),
    expected_size=9_601_187_887,
)

MODEL_SHA256 = "6cafee55eaf087a93b6e604d072da459c6200b15616f14c32120e29f32be9852"
HISTORICAL_CONFIG_SHA256 = "4ca5c8bdcdafdc55e4ef4cf5cb96ca2747498ac15752bf45eecff66c4f3456a2"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def quarantine(path: Path, reason: str) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target = path.with_name(f"{path.name}.bad-{reason}-{stamp}")
    counter = 1
    while target.exists():
        target = path.with_name(f"{path.name}.bad-{reason}-{stamp}-{counter}")
        counter += 1
    path.rename(target)
    return target


def download_with_resume(artifact: Artifact, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) == artifact.sha256:
            print(f"verified existing download: {destination}")
            return
        moved = quarantine(destination, "checksum")
        print(f"quarantined checksum mismatch: {moved}")

    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 6):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "MobiWAM-bootstrap/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(artifact.url, headers=headers)

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status_code = getattr(response, "status", response.getcode())
                if offset and status_code != 206:
                    moved = quarantine(partial, "no-range")
                    print(f"server ignored resume; retained old partial at {moved}")
                    offset = 0
                mode = "ab" if offset and status_code == 206 else "wb"
                with partial.open(mode) as output:
                    copied = offset
                    last_report = time.monotonic()
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        copied += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 30:
                            print(f"{artifact.filename}: {copied / 2**30:.2f} GiB downloaded")
                            last_report = now
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and partial.exists() and sha256_file(partial) == artifact.sha256:
                os.replace(partial, destination)
                print(f"completed partial verified after HTTP 416: {destination}")
                return
            print(f"download attempt {attempt}/5 failed: HTTP {exc.code}: {exc.reason}")
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 30))
            continue
        except (OSError, urllib.error.URLError) as exc:
            print(f"download attempt {attempt}/5 failed: {type(exc).__name__}: {exc}")
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 30))
            continue

        actual_hash = sha256_file(partial)
        if actual_hash == artifact.sha256:
            os.replace(partial, destination)
            print(f"download verified: {destination}")
            return

        moved = quarantine(partial, "checksum")
        print(f"download checksum mismatch; retained at {moved}")
        if attempt == 5:
            raise RuntimeError(f"checksum mismatch for {artifact.filename}: {actual_hash}")

    raise RuntimeError(f"download failed: {artifact.filename}")


def validate_zip_members(archive: zipfile.ZipFile) -> None:
    for info in archive.infolist():
        member = PurePosixPath(info.filename)
        if member.is_absolute() or ".." in member.parts:
            raise RuntimeError(f"unsafe ZIP path: {info.filename}")
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"symbolic links are not accepted in ZIP: {info.filename}")


def extract_artifact(artifact: Artifact, archive_path: Path, root: Path) -> Path:
    final_root = root / artifact.extract_root
    expected_path = final_root / artifact.expected_relative_path
    if expected_path.is_file():
        if expected_path.stat().st_size != artifact.expected_size:
            raise RuntimeError(f"existing artifact has unexpected size: {expected_path}")
        print(f"verified existing extraction: {expected_path}")
        return expected_path

    final_root.mkdir(parents=True, exist_ok=True)
    staging_parent = root / "downloads" / "mobipi" / "staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{artifact.filename}.", dir=staging_parent))
    completed = False
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            validate_zip_members(archive)
            bad_member = archive.testzip()
            if bad_member is not None:
                raise RuntimeError(f"ZIP CRC failure: {bad_member}")
            archive.extractall(staging)

        staged_expected = staging / artifact.expected_relative_path
        if not staged_expected.is_file():
            raise FileNotFoundError(f"expected file missing from ZIP: {staged_expected}")
        if staged_expected.stat().st_size != artifact.expected_size:
            raise RuntimeError(f"unexpected extracted size: {staged_expected}")

        top_component = PurePosixPath(artifact.expected_relative_path).parts[0]
        staged_top = staging / top_component
        final_top = final_root / top_component
        if final_top.exists():
            raise RuntimeError(
                f"refusing to merge into an existing partial tree: {final_top}"
            )
        shutil.move(str(staged_top), str(final_top))
        completed = True
    finally:
        if completed:
            shutil.rmtree(staging)
        else:
            print(f"retained failed extraction for inspection: {staging}")

    if not expected_path.is_file():
        raise FileNotFoundError(expected_path)
    return expected_path


def patch_checkpoint_config(root: Path) -> dict[str, str]:
    checkpoint_dir = (
        root
        / CHECKPOINT.extract_root
        / "04-12-CloseSingleDoor"
        / "seed_1_CloseSingleDoor_mg-300"
        / "20250413055045"
    )
    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    before_hash = sha256_file(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_dataset = root / DATASET.extract_root / DATASET.expected_relative_path
    config["train"]["data"][0]["path"] = str(expected_dataset)
    rendered = json.dumps(config, indent=2) + "\n"

    if config_path.read_text(encoding="utf-8") != rendered:
        temporary = config_path.with_name(f".{config_path.name}.tmp.{os.getpid()}")
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, config_path)

    return {
        "path": str(config_path),
        "sha256_before_path_rewrite": before_hash,
        "sha256_after_path_rewrite": sha256_file(config_path),
        "historical_other_machine_sha256": HISTORICAL_CONFIG_SHA256,
        "dataset_path": str(expected_dataset),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--with-dataset",
        action="store_true",
        help="also download the 5.6 GB source ZIP required by unmodified upstream evaluation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    allowed_root = Path("/share/chensiyu").resolve()
    if allowed_root not in root.parents:
        raise RuntimeError(f"Project root must stay under {allowed_root}: {root}")

    downloads = root / "downloads" / "mobipi"
    audit: dict[str, object] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root),
        "artifacts": {},
    }

    selected = [CHECKPOINT]
    if args.with_dataset:
        selected.append(DATASET)

    for artifact in selected:
        archive_path = downloads / artifact.filename
        download_with_resume(artifact, archive_path)
        expected_path = extract_artifact(artifact, archive_path, root)
        artifact_record: dict[str, object] = {
            "url": artifact.url,
            "archive": str(archive_path),
            "archive_sha256": sha256_file(archive_path),
            "expected_file": str(expected_path),
            "expected_file_size": expected_path.stat().st_size,
        }
        if artifact is CHECKPOINT:
            actual_model_hash = sha256_file(expected_path)
            if actual_model_hash != MODEL_SHA256:
                raise RuntimeError(f"checkpoint model checksum mismatch: {actual_model_hash}")
            artifact_record["model_sha256"] = actual_model_hash
        audit["artifacts"][artifact.filename] = artifact_record

    audit["config"] = patch_checkpoint_config(root)
    audit_dir = root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "mobipi_close_single_door_artifacts.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"artifact audit written: {audit_path}")
    if not args.with_dataset:
        print("checkpoint ready; rerun with --with-dataset before unmodified vanilla evaluation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
