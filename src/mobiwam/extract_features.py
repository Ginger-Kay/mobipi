from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .feature_cache import FeatureCache, feature_cache_key
from .records import DataSplit, RouteRolloutRecord, RouteType, SourceStateRecord, Stage


VISUAL_KEYS = (
    "robot0_agentview_left_image",
    "robot0_agentview_right_image",
    "robot0_eye_in_hand_image",
)
EXCLUDED_OBSERVABLE_KEYS = frozenset({"object"})
OPTION_IDS = {RouteType.EXECUTE: 0, RouteType.DOCK: 1, RouteType.ASSIST: 2}
EVENT_IDS = {
    "QUERY": 0,
    "EXECUTE": 1,
    "REPLAN": 2,
    "MOVE": 3,
    "SETTLE": 4,
    "OBSERVE": 5,
    "HISTORY_RESET": 6,
    "ASSIST": 7,
    "POST_DOCK_POLICY_READY": 8,
    "TERMINAL": 10,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(content).hexdigest()


def load_candidate_grid(path: Path) -> dict[tuple[RouteType, str], dict[str, Any]]:
    grid = json.loads(path.read_text())
    rows = {
        RouteType.EXECUTE: grid.get(
            "execute_candidates",
            [{"candidate_id": "e0", "candidate_params": {"base_locked": True}}],
        ),
        RouteType.DOCK: grid["dock_candidates"],
        RouteType.ASSIST: grid["assist_candidates"],
    }
    result: dict[tuple[RouteType, str], dict[str, Any]] = {}
    for route, candidates in rows.items():
        for candidate in candidates:
            key = (route, str(candidate["candidate_id"]))
            if key in result:
                raise ValueError(f"duplicate frozen candidate: {route.value}/{key[1]}")
            result[key] = dict(candidate.get("candidate_params", {}))
    return result


def _xy(value: object) -> tuple[float, float]:
    if value is None:
        return 0.0, 0.0
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError("candidate XY value must contain two finite numbers")
    return float(array[0]), float(array[1])


def encode_candidate_params(
    route: RouteType, params: Mapping[str, Any]
) -> np.ndarray:
    """Encode only frozen selection-time candidate fields into 16 values."""

    vector = np.zeros(16, dtype=np.float32)
    vector[OPTION_IDS[route]] = 1.0
    vector[3] = float(bool(params.get("base_locked", False)))
    target_x, target_y = _xy(params.get("target_offset_local_xy_m"))
    delta = params.get(
        "waypoint_delta_from_start_dock_xy_m",
        params.get("target_delta_from_start_dock_xy_m"),
    )
    delta_x, delta_y = _xy(delta)
    vector[4:8] = (target_x, target_y, delta_x, delta_y)
    vector[8] = float(params.get("dock_max_steps", 0)) / 500.0
    vector[9] = float(params.get("waypoint_max_steps", 0)) / 500.0
    vector[10] = float(params.get("position_tolerance_m", 0.0)) / 0.05
    vector[11] = float(params.get("yaw_tolerance_rad", 0.0)) / np.pi
    vector[12] = float(params.get("command_gain", 0.0))
    vector[13] = float(params.get("fraction_toward_dock", 0.0))
    vector[14] = float(params.get("max_translation_m", 0.0)) / 0.20
    vector[15] = float(params.get("max_yaw_rad", 0.0)) / np.pi
    if not np.all(np.isfinite(vector)):
        raise ValueError("candidate encoding contains non-finite values")
    return vector


def planned_duration(route: RouteType, params: Mapping[str, Any]) -> float:
    if route is RouteType.EXECUTE:
        return 1.0
    if route is RouteType.DOCK:
        steps = float(params.get("dock_max_steps", 0)) + float(
            params.get("waypoint_max_steps", 0)
            if "waypoint_delta_from_start_dock_xy_m" in params
            else 0
        )
        return steps / 500.0
    return float(params.get("fraction_toward_dock", 0.0)) + float(
        params.get("max_translation_m", 0.0)
    )


def observable_proprio_token(history: Mapping[str, np.ndarray]) -> np.ndarray:
    values: list[np.ndarray] = []
    for key in sorted(history):
        if key in VISUAL_KEYS or key in EXCLUDED_OBSERVABLE_KEYS:
            continue
        if not (
            key.startswith("robot0_")
            or key in {"actions", "timesteps"}
        ):
            continue
        array = np.asarray(history[key], dtype=np.float64)
        if array.ndim < 2 or not np.all(np.isfinite(array)):
            raise ValueError(f"invalid observable history array: {key}")
        flat = array.reshape(array.shape[0], -1)
        values.extend((flat[-1], flat.mean(axis=0), flat.std(axis=0)))
    if not values:
        raise ValueError("no approved observable proprio/history keys found")
    raw = np.concatenate(values)
    token = np.zeros(1024, dtype=np.float32)
    count = min(len(raw), len(token))
    token[:count] = np.tanh(raw[:count]).astype(np.float32)
    return token


class FrozenCLIPVisionEncoder:
    def __init__(self, model_path: Path, *, device: str, batch_size: int):
        import torch
        from transformers import CLIPVisionModel

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model = CLIPVisionModel.from_pretrained(
            str(model_path), local_files_only=True
        ).to(self.device).eval()
        self.model.requires_grad_(False)
        if int(self.model.config.hidden_size) != 1024:
            raise RuntimeError("frozen CLIP vision hidden size must be 1024")
        self.mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], device=self.device
        ).reshape(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], device=self.device
        ).reshape(1, 3, 1, 1)

    def __call__(self, images: np.ndarray) -> np.ndarray:
        torch = self.torch
        array = np.asarray(images, dtype=np.float32)
        if array.ndim != 4 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
            raise ValueError("visual history must have shape [time, 3, H, W]")
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(array), self.batch_size):
                batch = torch.as_tensor(
                    array[start : start + self.batch_size], device=self.device
                )
                batch = torch.nn.functional.interpolate(
                    batch,
                    size=(224, 224),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
                batch = (batch - self.mean) / self.std
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ):
                    encoded = self.model(pixel_values=batch).pooler_output
                outputs.append(encoded.float().cpu().numpy())
        return np.concatenate(outputs, axis=0)


def encode_source_context(
    history_path: Path,
    encoder: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    with np.load(history_path, allow_pickle=False) as archive:
        history = {name: archive[name] for name in archive.files}
    tokens = []
    for key in VISUAL_KEYS:
        if key not in history:
            raise ValueError(f"observable history lacks frozen view: {key}")
        encoded = np.asarray(encoder(history[key]), dtype=np.float32)
        if encoded.shape != (len(history[key]), 1024):
            raise ValueError(f"encoder returned wrong shape for {key}: {encoded.shape}")
        token = encoded.mean(axis=0)
        norm = float(np.linalg.norm(token))
        tokens.append(token / max(norm, 1e-12))
    tokens.append(observable_proprio_token(history))
    result = np.stack(tokens).astype(np.float32)
    if result.shape != (4, 1024) or not np.all(np.isfinite(result)):
        raise ValueError("source context must be finite with shape [4, 1024]")
    return result


def _rotation_matrix_to_xyzw(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array([0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[2, 1] - matrix[1, 2]) / scale])
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array([(matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale])
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array([(matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale, (matrix[1, 0] - matrix[0, 1]) / scale])
    return quat / max(float(np.linalg.norm(quat)), 1e-12)


def resample_induced_trajectory(path: Path, horizon: int = 16) -> tuple[np.ndarray, float]:
    with np.load(path, allow_pickle=False) as archive:
        poses = np.asarray(archive["eef_poses"], dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses):
        return np.zeros((horizon, 7), dtype=np.float32), 0.0
    indices = np.rint(np.linspace(0, len(poses) - 1, horizon)).astype(int)
    selected = poses[indices]
    initial = poses[0]
    result = np.zeros((horizon, 7), dtype=np.float32)
    result[:, :3] = (selected[:, :3, 3] - initial[:3, 3]).astype(np.float32)
    for index, pose in enumerate(selected):
        relative = initial[:3, :3].T @ pose[:3, :3]
        result[index, 3:] = _rotation_matrix_to_xyzw(relative).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"trajectory contains non-finite values: {path}")
    return result, 1.0


def _event_targets(row: RouteRolloutRecord) -> tuple[np.ndarray, np.ndarray]:
    typed = np.zeros((3, 512), dtype=np.float32)
    route_events = {
        RouteType.EXECUTE: ("QUERY", "EXECUTE", "REPLAN"),
        RouteType.DOCK: ("MOVE", "POST_DOCK_POLICY_READY", "EXECUTE"),
        RouteType.ASSIST: ("QUERY", "ASSIST", "REPLAN"),
    }[row.route_type]
    risk = float(row.irreversible_failure or row.collision)
    numeric = np.asarray(
        [
            row.stage_eligible,
            row.hard_valid,
            row.success,
            risk,
            row.collision,
            row.contact_loss,
            row.progress_delta,
            row.execution_time_s / 500.0,
            row.base_path_length_m / 2.0,
            row.intent_pos_error_p95_m or 0.0,
            row.intent_rot_error_p95_rad or 0.0,
        ],
        dtype=np.float32,
    )
    for index, event in enumerate(route_events):
        typed[index, EVENT_IDS[event]] = 1.0
        typed[index, 32 : 32 + len(numeric)] = numeric
    boundary = np.zeros(512, dtype=np.float32)
    boundary[: len(numeric)] = numeric
    return typed, boundary


def _duration_cost(row: RouteRolloutRecord) -> np.ndarray:
    return np.asarray(
        [
            row.execution_time_s / 500.0,
            row.base_path_length_m / 2.0,
            float(row.collision),
            float(row.contact_loss),
            float(row.intent_pos_error_p95_m or 0.0),
            row.progress_delta,
        ],
        dtype=np.float32,
    )


def assemble_feature_arrays(
    sources: Sequence[SourceStateRecord],
    rollouts: Sequence[RouteRolloutRecord],
    *,
    candidate_lookup: Mapping[tuple[RouteType, str], Mapping[str, Any]],
    contexts: Mapping[str, np.ndarray],
    allowed_splits: frozenset[DataSplit],
    add_a0_audit: bool = True,
) -> dict[str, np.ndarray]:
    source_by_id = {source.source_state_id: source for source in sources}
    if len(source_by_id) != len(sources):
        raise ValueError("duplicate source-state row")
    if DataSplit.LOCKED_TEST in allowed_splits or DataSplit.TEST in allowed_splits:
        raise ValueError("train-side extraction cannot allow locked/test splits")
    for source in sources:
        if source.split not in allowed_splits:
            raise ValueError(
                f"source {source.source_state_id} split {source.split.value} is not allowed"
            )
        if source.source_state_id not in contexts:
            raise ValueError(f"missing source context: {source.source_state_id}")

    rows: list[dict[str, Any]] = []
    first_execute: dict[str, dict[str, Any]] = {}
    for rollout in rollouts:
        source = source_by_id.get(rollout.source_state_id)
        if source is None:
            raise ValueError(f"rollout references unknown source: {rollout.source_state_id}")
        if rollout.split is not source.split:
            raise ValueError("rollout/source split mismatch")
        key = (rollout.route_type, rollout.candidate_id)
        if key not in candidate_lookup:
            raise ValueError(f"rollout candidate absent from frozen grid: {key}")
        frozen_params = candidate_lookup[key]
        trajectory, trajectory_valid = resample_induced_trajectory(
            Path(rollout.state_trace_path)
        )
        typed, boundary = _event_targets(rollout)
        row = {
            "context": np.asarray(contexts[rollout.source_state_id], dtype=np.float32),
            "option_id": OPTION_IDS[rollout.route_type],
            "candidate_params": encode_candidate_params(rollout.route_type, frozen_params),
            "phase_id": 0 if rollout.stage is Stage.PRECONTACT else 1,
            "duration": planned_duration(rollout.route_type, frozen_params),
            "success": float(rollout.success),
            "risk": float(rollout.irreversible_failure or rollout.collision),
            "duration_cost": _duration_cost(rollout),
            "source_id": rollout.source_state_id,
            "split": rollout.split.value,
            "typed": typed,
            "boundary": boundary,
            "is_a0": False,
            "trajectory": trajectory,
            "trajectory_valid": trajectory_valid,
            "task_id": rollout.task_id,
            "task_family": rollout.task_family,
            "route": rollout.route_type.value,
            "candidate_id": rollout.candidate_id,
            "repeat_index": rollout.repeat_index,
        }
        rows.append(row)
        if rollout.route_type is RouteType.EXECUTE:
            previous = first_execute.get(rollout.source_state_id)
            if previous is None or rollout.repeat_index < previous["repeat_index"]:
                first_execute[rollout.source_state_id] = row

    if add_a0_audit:
        zero_params = {
            "target_offset_local_xy_m": [0.0, 0.0],
            "fraction_toward_dock": 0.0,
            "max_translation_m": 0.0,
            "max_yaw_rad": 0.0,
        }
        for source_id in sorted(source_by_id):
            if source_id not in first_execute:
                raise ValueError(f"source lacks E row for A(0)=E audit: {source_id}")
            execute = first_execute[source_id]
            audit = dict(execute)
            audit["option_id"] = OPTION_IDS[RouteType.ASSIST]
            audit["candidate_params"] = encode_candidate_params(
                RouteType.ASSIST, zero_params
            )
            audit["duration"] = 0.0
            audit["is_a0"] = True
            audit["route"] = "A0"
            audit["candidate_id"] = "a_zero_consistency_audit"
            audit["repeat_index"] = 0
            rows.append(audit)

    if not rows:
        raise ValueError("feature extraction produced no rows")
    arrays = {
        "context": np.stack([row["context"] for row in rows]).astype(np.float16),
        "option_ids": np.asarray([row["option_id"] for row in rows], dtype=np.int64),
        "candidate_params": np.stack([row["candidate_params"] for row in rows]),
        "phase_ids": np.asarray([row["phase_id"] for row in rows], dtype=np.int64),
        "duration": np.asarray([row["duration"] for row in rows], dtype=np.float32),
        "success": np.asarray([row["success"] for row in rows], dtype=np.float32),
        "irreversible_risk": np.asarray([row["risk"] for row in rows], dtype=np.float32),
        "duration_cost": np.stack([row["duration_cost"] for row in rows]),
        "source_ids": np.asarray([row["source_id"] for row in rows]),
        "split": np.asarray([row["split"] for row in rows]),
        "typed_internal_states": np.stack([row["typed"] for row in rows]),
        "common_boundary_latent": np.stack([row["boundary"] for row in rows]),
        "is_a0": np.asarray([row["is_a0"] for row in rows], dtype=np.bool_),
        "induced_ee_trajectory": np.stack([row["trajectory"] for row in rows]),
        "trajectory_valid": np.asarray([row["trajectory_valid"] for row in rows], dtype=np.float32),
        "task_ids": np.asarray([row["task_id"] for row in rows]),
        "task_families": np.asarray([row["task_family"] for row in rows]),
        "route_types": np.asarray([row["route"] for row in rows]),
        "candidate_ids": np.asarray([row["candidate_id"] for row in rows]),
        "repeat_indices": np.asarray([row["repeat_index"] for row in rows], dtype=np.int64),
    }
    count = len(rows)
    if any(len(value) != count for value in arrays.values()):
        raise RuntimeError("feature arrays have inconsistent row counts")
    for name, value in arrays.items():
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError(f"feature array contains non-finite values: {name}")
    return arrays


def _load_jsonl(path: Path, factory: Any) -> list[Any]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(factory.from_mapping(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frozen observable train-side MMWAM features"
    )
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--candidate-grid", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path, required=True)
    parser.add_argument("--encoder-revision", required=True)
    parser.add_argument("--expected-encoder-weight-sha256", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--allowed-split",
        action="append",
        choices=["train", "validation", "calibration"],
    )
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite feature output or manifest")
    encoder_weight = args.encoder_path / "pytorch_model.bin"
    observed_encoder_weight_sha256 = sha256_file(encoder_weight)
    if observed_encoder_weight_sha256 != args.expected_encoder_weight_sha256:
        raise RuntimeError(
            "frozen encoder weight checksum differs from the expected checksum"
        )
    allowed = frozenset(
        DataSplit(value)
        for value in (args.allowed_split or ["train", "validation", "calibration"])
    )
    sources = _load_jsonl(args.sources, SourceStateRecord)
    rollouts = _load_jsonl(args.rollouts, RouteRolloutRecord)
    if any(source.split not in allowed for source in sources):
        raise ValueError("input source file contains a disallowed or locked split")
    if any(row.split not in allowed for row in rollouts):
        raise ValueError("input rollout file contains a disallowed or locked split")

    candidate_lookup = load_candidate_grid(args.candidate_grid)
    observable_spec = {
        "visual_keys": VISUAL_KEYS,
        "excluded_keys": sorted(EXCLUDED_OBSERVABLE_KEYS),
        "context_shape": [4, 1024],
        "preprocess": "bicubic224-clip-mean-std-temporal-mean-v1",
    }
    spec_checksum = stable_hash(observable_spec)
    cache = FeatureCache(args.cache_root)
    encoder = FrozenCLIPVisionEncoder(
        args.encoder_path, device=args.device, batch_size=args.batch_size
    )
    contexts: dict[str, np.ndarray] = {}
    cache_records: dict[str, dict[str, object]] = {}
    for source in sources:
        key = feature_cache_key(
            source_checksum=source.snapshot_hash,
            candidate_checksum=spec_checksum,
            encoder_revision=(
                f"{args.encoder_revision}:{observed_encoder_weight_sha256}"
            ),
        )
        path = cache.path_for(key)
        if path.is_file():
            context = cache.get(key)
        else:
            context = encode_source_context(
                Path(source.snapshot_path) / "frame_history.npz", encoder
            )
            cache.put(key, context.astype(np.float16))
        if context.shape != (4, 1024):
            raise ValueError(f"cached source context has wrong shape: {source.source_state_id}")
        contexts[source.source_state_id] = context
        cache_records[source.source_state_id] = {
            "key": key,
            "path": str(path),
            "sha256": sha256_file(path),
        }

    arrays = assemble_feature_arrays(
        sources,
        rollouts,
        candidate_lookup=candidate_lookup,
        contexts=contexts,
        allowed_splits=allowed,
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
        "data_role": "train_validation_calibration_only",
        "locked_test_opened": False,
        "sources": str(args.sources),
        "sources_sha256": sha256_file(args.sources),
        "rollouts": str(args.rollouts),
        "rollouts_sha256": sha256_file(args.rollouts),
        "candidate_grid": str(args.candidate_grid),
        "candidate_grid_sha256": sha256_file(args.candidate_grid),
        "encoder_path": str(args.encoder_path),
        "encoder_revision": args.encoder_revision,
        "encoder_weight": str(encoder_weight),
        "encoder_weight_sha256": observed_encoder_weight_sha256,
        "observable_spec": observable_spec,
        "observable_spec_checksum": spec_checksum,
        "allowed_splits": sorted(split.value for split in allowed),
        "source_count": len(sources),
        "rollout_record_count": len(rollouts),
        "feature_row_count": len(arrays["source_ids"]),
        "a0_auxiliary_rows": int(arrays["is_a0"].sum()),
        "split_counts": dict(sorted(Counter(arrays["split"].tolist()).items())),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "cache_records": cache_records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "pass", "rows": manifest["feature_row_count"]}))


if __name__ == "__main__":
    main()
