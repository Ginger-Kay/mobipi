#!/usr/bin/env python3
"""CPU-only environment and artifact preflight for the pinned Mobi-pi stack."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/share/chensiyu/MobiWAM")
ROOT_COMMIT = "19b130b8ada3f7e029918449c12d433e9e629ca1"
SUBMODULE_COMMITS = {
    "external/diffusion_policy": "5ba07ac6661db573af695b419a7947ecb704690f",
    "external/lelan": "8a84208be913b3838f2e550929d39cd0d674b252",
    "external/mimicgen": "05c723bdf9b4e1ddb77fea63bdde21920408b5fd",
    "external/robocasa": "3683fb01d6fc87d7849e7b5a886d01f4b1d7a55d",
    "external/robomimic": "ff9f7f4157a5c8257f17c5910067030e2291378f",
    "external/robomimic/act": "742c753c0d4a5d87076c8f69e5628c79a8cc5488",
}
EXPECTED_DISTRIBUTIONS = {
    "numpy": "1.23.3",
    "torch": "2.2.0",
    "torchvision": "0.17.0",
    "torchaudio": "2.2.0",
    "robosuite": "1.5.0",
    "robocasa": "0.2.0",
    "robomimic": "0.3.0",
    "mimicgen": "1.0.0",
    "mobipi": "0.1.0",
    "mujoco": "3.2.6",
    "opencv-python": "4.6.0.66",
    "open3d": "0.19.0",
    "transformers": "4.36.0",
    "diffusers": "0.11.1",
    "huggingface-hub": "0.25.0",
    "timm": "1.0.12",
}
MODEL_RELATIVE_PATH = Path(
    "checkpoints/robocasa/bc_xfmr/04-12-CloseSingleDoor/"
    "seed_1_CloseSingleDoor_mg-300/20250413055045/models/model_epoch_1000.pth"
)
MODEL_SIZE = 246_511_905
MODEL_SHA256 = "6cafee55eaf087a93b6e604d072da459c6200b15616f14c32120e29f32be9852"
DATASET_RELATIVE_PATH = Path(
    "data/v0.1/single_stage/kitchen_doors/CloseSingleDoor/mg/"
    "2024-05-04-22-34-56/demo_im128_fixview.hdf5"
)
DATASET_SIZE = 9_601_187_887
CLIP_SIZE = 1_710_671_599
CLIP_SHA256 = "f1a17cdbe0f36fec524f5cafb1c261ea3bbbc13e346e0f74fc9eb0460dedd0d3"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def path_is_within(path: Path, parents: tuple[Path, ...]) -> bool:
    resolved = path.resolve()
    return any(resolved == parent or parent in resolved.parents for parent in parents)


def compatible_distribution_version(actual: str | None, expected: str) -> bool:
    """Accept PyTorch's CUDA wheel local tag while retaining the release pin."""
    return actual is not None and actual.split("+", 1)[0] == expected


def stage_requires_dataset(stage: str) -> bool:
    if stage not in {"environment", "ready", "full"}:
        raise ValueError(f"unknown preflight stage: {stage}")
    return stage == "full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--stage", choices=["environment", "ready", "full"], default="ready"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    allowed_root = Path("/share/chensiyu").resolve()
    if allowed_root not in root.parents:
        raise RuntimeError(f"Project root must stay under {allowed_root}: {root}")

    env_prefix = root / "envs" / "mobipi"
    repo = root / "repos" / "mobipi"
    output = args.output or root / "audit" / f"mobipi_{args.stage}_preflight.json"
    errors: list[str] = []
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": args.stage,
        "root": str(root),
        "python": {
            "executable": sys.executable,
            "prefix": sys.prefix,
            "version": sys.version,
            "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }

    if Path(sys.prefix).resolve() != env_prefix.resolve():
        errors.append(f"wrong Python prefix: {sys.prefix}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        errors.append("PYTHONNOUSERSITE must be 1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        errors.append("CPU preflight requires CUDA_VISIBLE_DEVICES to be empty")

    # Shared development images may lack libGL. The bootstrap puts it in this
    # dedicated Conda environment; make it visible before native imports.
    env_lib = env_prefix / "lib"
    if env_lib.is_dir():
        inherited_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = (
            f"{env_lib}:{inherited_library_path}" if inherited_library_path else str(env_lib)
        )

    source_roots = (env_prefix.resolve(), repo.resolve())
    module_names = [
        "numpy",
        "torch",
        "torchvision",
        "torchaudio",
        "cv2",
        "h5py",
        "mujoco",
        "robosuite",
        "robocasa",
        "robomimic",
        "mimicgen",
        "mobipi",
        "open3d",
        "transformers",
        "diffusers",
        "mobipi.eval.eval_baseline",
    ]
    modules: dict[str, dict[str, Any]] = {}
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            source = getattr(module, "__file__", None)
            modules[module_name] = {
                "source": source,
                "version": getattr(module, "__version__", None),
            }
            if source is not None and not path_is_within(Path(source), source_roots):
                errors.append(f"module escaped project environment: {module_name} -> {source}")
        except Exception as exc:  # noqa: BLE001 - all import failures belong in the audit
            modules[module_name] = {"error": f"{type(exc).__name__}: {exc}"}
            errors.append(f"failed to import {module_name}: {type(exc).__name__}: {exc}")
    report["modules"] = modules

    distributions: dict[str, str | None] = {}
    for name, expected in EXPECTED_DISTRIBUTIONS.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        distributions[name] = actual
        if not compatible_distribution_version(actual, expected):
            errors.append(f"distribution mismatch: {name} expected {expected}, got {actual}")
    report["distributions"] = distributions

    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        report["torch_runtime"] = {
            "version": torch_module.__version__,
            "compiled_cuda": torch_module.version.cuda,
            "cuda_available_with_devices_hidden": torch_module.cuda.is_available(),
        }
        if torch_module.version.cuda != "12.1":
            errors.append(f"Torch CUDA runtime mismatch: {torch_module.version.cuda}")
        if torch_module.cuda.is_available():
            errors.append("CPU preflight unexpectedly exposed a CUDA device")

    commits: dict[str, str] = {}
    try:
        commits["root"] = git_head(repo)
        if commits["root"] != ROOT_COMMIT:
            errors.append(f"Mobi-pi commit mismatch: {commits['root']}")
        for relative_path, expected in SUBMODULE_COMMITS.items():
            actual = git_head(repo / relative_path)
            commits[relative_path] = actual
            if actual != expected:
                errors.append(f"submodule mismatch: {relative_path} -> {actual}")
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"source commit check failed: {type(exc).__name__}: {exc}")
    report["commits"] = commits

    try:
        from mobipi import macros as mobipi_macros
        from robocasa import macros as robocasa_macros
        from robomimic import macros as robomimic_macros

        macros = {
            "SCENE_MODEL_ROOT_DIR": mobipi_macros.SCENE_MODEL_ROOT_DIR,
            "POLICY_CKPT_ROOT_DIR": mobipi_macros.POLICY_CKPT_ROOT_DIR,
            "LOG_ROOT_DIR": mobipi_macros.LOG_ROOT_DIR,
            "DATA_ROOT_DIR": mobipi_macros.DATA_ROOT_DIR,
            "ROBOCASA_DATASET_BASE_PATH": robocasa_macros.DATASET_BASE_PATH,
            "ROBOMIMIC_EXPDATA_BASE_PATH": robomimic_macros.EXPDATA_BASE_PATH,
        }
        expected_macros = {
            "SCENE_MODEL_ROOT_DIR": str(root / "assets" / "scene_models"),
            "POLICY_CKPT_ROOT_DIR": str(root / "checkpoints"),
            "LOG_ROOT_DIR": str(root / "experiments" / "mobipi"),
            "DATA_ROOT_DIR": str(root / "data"),
            "ROBOCASA_DATASET_BASE_PATH": str(root / "data"),
            "ROBOMIMIC_EXPDATA_BASE_PATH": str(root / "experiments" / "robomimic"),
        }
        report["macros"] = macros
        for name, expected in expected_macros.items():
            if macros.get(name) != expected:
                errors.append(f"macro mismatch: {name} -> {macros.get(name)}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"private macro check failed: {type(exc).__name__}: {exc}")

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    pip_lines = [line for line in pip_check.stdout.splitlines() if line.strip()]
    allowed_conflict_prefixes = (
        "robomimic 0.3.0 has requirement numpy==1.23.2",
        "robomimic 0.3.0 has requirement torch==2.0.1",
        "robomimic 0.3.0 has requirement torchvision==0.15.2",
    )
    unexpected_conflicts = [
        line for line in pip_lines if not line.startswith(allowed_conflict_prefixes)
    ]
    report["pip_check"] = {
        "exit_code": pip_check.returncode,
        "lines": pip_lines,
        "allowed_upstream_metadata_conflicts": list(allowed_conflict_prefixes),
    }
    if unexpected_conflicts:
        errors.extend(f"unexpected pip check conflict: {line}" for line in unexpected_conflicts)

    if args.stage in ("ready", "full"):
        artifacts: dict[str, Any] = {}
        model = root / MODEL_RELATIVE_PATH
        if not model.is_file():
            errors.append(f"checkpoint missing: {model}")
        else:
            model_hash = sha256_file(model)
            artifacts["checkpoint"] = {
                "path": str(model),
                "size": model.stat().st_size,
                "sha256": model_hash,
            }
            if model.stat().st_size != MODEL_SIZE or model_hash != MODEL_SHA256:
                errors.append("checkpoint integrity check failed")

        if stage_requires_dataset(args.stage):
            dataset = root / DATASET_RELATIVE_PATH
            if not dataset.is_file():
                errors.append(f"dataset missing: {dataset}")
            else:
                dataset_record: dict[str, Any] = {
                    "path": str(dataset),
                    "size": dataset.stat().st_size,
                }
                if dataset.stat().st_size != DATASET_SIZE:
                    errors.append(f"dataset size mismatch: {dataset.stat().st_size}")
                try:
                    import h5py

                    with h5py.File(dataset, "r") as handle:
                        dataset_record["top_level_keys"] = sorted(handle.keys())
                        dataset_record["demo_count"] = len(handle.get("data", {}))
                        dataset_record["filter_300_demos_count"] = len(handle["mask"]["300_demos"])
                        if dataset_record["filter_300_demos_count"] != 300:
                            errors.append("dataset filter 300_demos does not contain 300 demos")
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"HDF5 metadata check failed: {type(exc).__name__}: {exc}")
                artifacts["dataset"] = dataset_record

        clip_audit = root / "audit" / "mobipi_clip_cache.json"
        if not clip_audit.is_file():
            errors.append(f"CLIP cache audit missing: {clip_audit}")
        else:
            clip_record = json.loads(clip_audit.read_text(encoding="utf-8"))
            clip_weight = Path(clip_record["weight_path"])
            if not clip_weight.is_file():
                errors.append(f"CLIP weight missing: {clip_weight}")
            else:
                clip_hash = sha256_file(clip_weight)
                if clip_weight.stat().st_size != CLIP_SIZE or clip_hash != CLIP_SHA256:
                    errors.append("CLIP weight integrity check failed")
                clip_record["verified_sha256"] = clip_hash
            artifacts["clip"] = clip_record

        asset_root = repo / "external" / "robocasa" / "robocasa" / "models" / "assets"
        asset_directories = [
            asset_root / "textures",
            asset_root / "fixtures",
            asset_root / "objects" / "objaverse",
            asset_root / "generative_textures",
        ]
        asset_status: dict[str, bool] = {}
        for directory in asset_directories:
            populated = directory.is_dir() and any(path.is_file() for path in directory.rglob("*"))
            asset_status[str(directory)] = populated
            if not populated:
                errors.append(f"RoboCasa asset directory is missing or empty: {directory}")
        artifacts["robocasa_assets"] = asset_status
        report["artifacts"] = artifacts

    report["errors"] = errors
    report["status"] = "pass" if not errors else "fail"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"preflight {report['status']}: {output}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
