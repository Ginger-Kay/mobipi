from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .models import EvaluatorConfig
from .training import build_model


LOCKED_FREEZE_FIELDS = frozenset(
    {
        "model_manifest_sha256",
        "candidate_grid_sha256",
        "source_schedule_checksum",
        "calibration_sha256",
        "threshold_config_sha256",
        "statistics_script_sha256",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_locked_freeze(path: Path) -> dict[str, Any]:
    freeze = json.loads(path.read_text())
    missing = sorted(LOCKED_FREEZE_FIELDS.difference(freeze))
    if freeze.get("status") != "frozen" or not freeze.get(
        "locked_test_open_authorized", False
    ):
        raise ValueError("locked-test freeze manifest is not frozen and authorized")
    if missing:
        raise ValueError(f"locked-test freeze manifest lacks fields: {missing}")
    for field in LOCKED_FREEZE_FIELDS:
        value = str(freeze[field])
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"locked-test freeze field is not a SHA-256 digest: {field}")
    return freeze


def prediction_arrays(
    model: torch.nn.Module,
    data: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    required = {
        "context",
        "option_ids",
        "candidate_params",
        "phase_ids",
        "duration",
        "source_ids",
        "split",
        "success",
        "irreversible_risk",
        "duration_cost",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"feature file is missing: {missing}")
    outputs: dict[str, list[np.ndarray]] = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            inputs = {
                "context": torch.as_tensor(
                    data["context"][batch_indices], device=device
                ),
                "option_ids": torch.as_tensor(
                    data["option_ids"][batch_indices], device=device
                ).long(),
                "candidate_params": torch.as_tensor(
                    data["candidate_params"][batch_indices], device=device
                ),
                "phase_ids": torch.as_tensor(
                    data["phase_ids"][batch_indices], device=device
                ).long(),
                "duration": torch.as_tensor(
                    data["duration"][batch_indices], device=device
                ),
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(**inputs)
            for name in (
                "success",
                "irreversible_risk",
                "duration_cost",
                "predictive_uncertainty",
                "induced_ee_trajectory",
            ):
                if name in prediction:
                    outputs.setdefault(name, []).append(
                        prediction[name].float().cpu().numpy()
                    )

    result = {
        "success_logits": np.concatenate(outputs["success"]),
        "risk_logits": np.concatenate(outputs["irreversible_risk"]),
        "duration_cost_prediction": np.concatenate(outputs["duration_cost"]),
        "success": np.asarray(data["success"])[indices],
        "irreversible_risk": np.asarray(data["irreversible_risk"])[indices],
        "duration_cost": np.asarray(data["duration_cost"])[indices],
        "source_ids": np.asarray(data["source_ids"])[indices],
        "split": np.asarray(data["split"])[indices],
    }
    for name in (
        "task_ids",
        "task_families",
        "route_types",
        "candidate_ids",
        "repeat_indices",
        "is_a0",
    ):
        if name in data:
            result[name] = np.asarray(data[name])[indices]
    if "predictive_uncertainty" in outputs:
        result["uncertainty_logits"] = np.concatenate(
            outputs["predictive_uncertainty"]
        )
    if "induced_ee_trajectory" in outputs:
        result["induced_ee_trajectory_prediction"] = np.concatenate(
            outputs["induced_ee_trajectory"]
        )
    count = len(indices)
    if any(len(value) != count for value in result.values()):
        raise RuntimeError("prediction arrays have inconsistent row counts")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export split-scoped MMWAM evaluator predictions"
    )
    parser.add_argument(
        "--model", choices=["value-only", "trajectory-only", "obc-wam"], required=True
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=["validation", "calibration", "locked_test"],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--locked-freeze-manifest", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite predictions or manifest")
    locked_freeze = None
    if args.split == "locked_test":
        if args.locked_freeze_manifest is None:
            parser.error("locked_test requires --locked-freeze-manifest")
        locked_freeze = validate_locked_freeze(args.locked_freeze_manifest)
    elif args.locked_freeze_manifest is not None:
        parser.error("locked freeze manifest is only valid with --split locked_test")

    with np.load(args.features, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    indices = np.flatnonzero(data["split"].astype(str) == args.split)
    if "is_a0" in data:
        indices = indices[~data["is_a0"][indices].astype(bool)]
    if not len(indices):
        raise ValueError(f"feature file has no rows for split {args.split}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("model_name") != args.model:
        raise ValueError("checkpoint model name differs from requested model")
    config = EvaluatorConfig(**checkpoint["config"])
    model = build_model(args.model, config).to(device)
    model.load_state_dict(checkpoint["model"])
    arrays = prediction_arrays(
        model,
        data,
        indices,
        device=device,
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(f".{args.output.name}.partial-{os.getpid()}")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, args.output)
    manifest = {
        "schema_version": "1.0",
        "status": "pass",
        "model": args.model,
        "split": args.split,
        "locked_test_opened": args.split == "locked_test",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "features": str(args.features),
        "features_sha256": sha256_file(args.features),
        "row_count": len(indices),
        "a0_auxiliary_rows_excluded": True,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "locked_freeze_manifest": (
            None
            if args.locked_freeze_manifest is None
            else str(args.locked_freeze_manifest)
        ),
        "locked_freeze_manifest_sha256": (
            None
            if args.locked_freeze_manifest is None
            else sha256_file(args.locked_freeze_manifest)
        ),
        "locked_freeze": locked_freeze,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "pass", "rows": len(indices), "split": args.split}))


if __name__ == "__main__":
    main()
