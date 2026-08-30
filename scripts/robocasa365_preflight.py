#!/usr/bin/env python3
"""CPU-only source and asset preflight for the isolated RoboCasa365 port."""

from __future__ import annotations

import argparse
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
ROBOCASA_COMMIT = "a07e365c958c4216cd6bbd5f30b47f09a65c6f00"
ROBOSUITE_COMMIT = "5ce6643f3092639d08f7b0f90ed1c6a84f50552c"


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def path_is_within(path: Path, parents: tuple[Path, ...]) -> bool:
    resolved = path.resolve()
    return any(resolved == parent or parent in resolved.parents for parent in parents)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stage", choices=["environment", "assets"], default="environment")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    allowed_root = Path("/share/chensiyu").resolve()
    if allowed_root not in root.parents:
        raise RuntimeError(f"Project root must stay under {allowed_root}: {root}")

    env_prefix = root / "envs" / "robocasa365"
    robocasa_repo = root / "repos" / "robocasa365"
    robosuite_repo = root / "repos" / "robosuite-robocasa365"
    output = args.output or root / "audit" / f"robocasa365_{args.stage}_preflight.json"
    errors: list[str] = []
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": args.stage,
        "root": str(root),
        "python": {"executable": sys.executable, "prefix": sys.prefix},
    }

    if Path(sys.prefix).resolve() != env_prefix.resolve():
        errors.append(f"wrong Python prefix: {sys.prefix}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        errors.append("PYTHONNOUSERSITE must be 1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        errors.append("CPU preflight requires CUDA_VISIBLE_DEVICES to be empty")

    expected_commits = {
        "robocasa": (robocasa_repo, ROBOCASA_COMMIT),
        "robosuite": (robosuite_repo, ROBOSUITE_COMMIT),
    }
    commits: dict[str, str] = {}
    for name, (path, expected) in expected_commits.items():
        try:
            commits[name] = git_head(path)
            if commits[name] != expected:
                errors.append(f"{name} commit mismatch: {commits[name]}")
            if subprocess.check_output(
                ["git", "-C", str(path), "status", "--short", "--untracked-files=no"],
                text=True,
            ).strip():
                errors.append(f"{name} has tracked working-tree changes")
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"{name} source check failed: {type(exc).__name__}: {exc}")
    report["commits"] = commits

    source_roots = (
        env_prefix.resolve(),
        robocasa_repo.resolve(),
        robosuite_repo.resolve(),
    )
    modules: dict[str, Any] = {}
    for name in ("numpy", "mujoco", "gymnasium", "robosuite", "robocasa"):
        try:
            module = importlib.import_module(name)
            source = getattr(module, "__file__", None)
            modules[name] = {
                "source": source,
                "version": getattr(module, "__version__", None),
            }
            if source is not None and not path_is_within(Path(source), source_roots):
                errors.append(f"module escaped isolated environment: {name} -> {source}")
        except Exception as exc:  # noqa: BLE001
            modules[name] = {"error": f"{type(exc).__name__}: {exc}"}
            errors.append(f"failed to import {name}: {type(exc).__name__}: {exc}")
    report["modules"] = modules

    distributions: dict[str, str | None] = {}
    for name in ("robocasa", "robosuite", "gymnasium", "mujoco"):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
            errors.append(f"distribution is missing: {name}")
    report["distributions"] = distributions
    if distributions.get("robocasa") != "1.0.1":
        errors.append(f"expected robocasa 1.0.1, got {distributions.get('robocasa')}")

    macros_private = robocasa_repo / "robocasa" / "macros_private.py"
    report["macros_private"] = str(macros_private)
    if not macros_private.is_file():
        errors.append(f"RoboCasa365 macros_private.py is missing: {macros_private}")

    if args.stage == "assets":
        asset_root = robocasa_repo / "robocasa" / "models" / "assets"
        relative_directories = (
            "textures",
            "generative_textures",
            "fixtures",
            "objects/objaverse",
            "objects/aigen_objs",
            "objects/lightwheel",
        )
        assets: dict[str, bool] = {}
        for relative in relative_directories:
            directory = asset_root / relative
            populated = directory.is_dir() and any(
                path.is_file() for path in directory.rglob("*")
            )
            assets[relative] = populated
            if not populated:
                errors.append(f"RoboCasa365 asset directory is empty: {directory}")
        report["assets"] = assets

    report["errors"] = errors
    report["status"] = "pass" if not errors else "fail"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"preflight {report['status']}: {output}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
