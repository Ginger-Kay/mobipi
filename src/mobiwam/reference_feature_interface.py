"""Development Source feature bridge; incomplete planner features fail closed.

Only time-zero observations enter context. Outcomes stay in a separate table.
This adapter does not certify the reference executor as the SCENE-004 planner.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

from .extract_features import VISUAL_KEYS, observable_proprio_token
from .scene004 import CANDIDATE_FEATURE_FIELDS, build_minimal_input, candidate_feature_vector


# Explicit sensors, not arbitrary robot0_* names or simulator state.
PROPRIO_KEYS = (
    "robot0_base_pos", "robot0_base_quat", "robot0_base_to_eef_pos",
    "robot0_base_to_eef_quat", "robot0_eef_pos", "robot0_eef_quat",
    "robot0_gripper_qpos", "robot0_gripper_qvel", "robot0_joint_pos_cos",
    "robot0_joint_pos_sin", "robot0_joint_vel",
)


def source_observations(path: Path) -> dict[str, np.ndarray]:
    """Read index zero only, before action zero; no episode attrs/outcomes."""
    result = {}
    with h5py.File(path, "r") as handle:
        obs = handle["data/demo_0/obs"]
        for key in (*VISUAL_KEYS, *PROPRIO_KEYS):
            value = np.asarray(obs[key][0:1])
            if value.shape[0] != 1 or not np.isfinite(value).all():
                raise ValueError(f"missing/invalid Source observation: {key}")
            if key in VISUAL_KEYS:
                if value.dtype != np.uint8 or value.ndim != 4 or value.shape[-1] != 3:
                    raise ValueError(f"Source RGB must be uint8 NHWC: {key}")
                value = value.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            elif value.ndim != 2:
                raise ValueError(f"Source proprio must be [1,D]: {key}")
            result[key] = value
    return result


def source_context(observations: Mapping[str, np.ndarray], encoder) -> np.ndarray:
    if set(observations) != set((*VISUAL_KEYS, *PROPRIO_KEYS)):
        raise ValueError("Source context accepts only the explicit sensor allowlist")
    if any(np.asarray(v).shape[0] != 1 for v in observations.values()):
        raise ValueError("Source context must contain exactly time zero")
    tokens = []
    for key in VISUAL_KEYS:
        encoded = np.asarray(encoder(observations[key]), dtype=np.float32)
        if encoded.shape != (1, 1024) or not np.isfinite(encoded).all():
            raise ValueError("frozen encoder must return finite [1,1024]")
        norm = float(np.linalg.norm(encoded[0]))
        if norm <= 1e-12:
            raise ValueError("zero visual embeddings are not valid features")
        tokens.append(encoded[0] / norm)
    tokens.append(observable_proprio_token(observations))
    context = np.stack(tokens).astype(np.float32)
    if not np.isfinite(context).all():
        raise ValueError("non-finite context")
    return context


def candidate_readiness(derived: Mapping[str, float]) -> dict:
    """Return a missing-feature report, never fill unknown planner values."""
    unknown = set(derived) - set(CANDIDATE_FEATURE_FIELDS)
    if unknown:
        raise ValueError(f"only schema-derived scalars accepted: {sorted(unknown)}")
    for key, value in derived.items():
        if not np.isscalar(value) or not np.isfinite(value):
            raise ValueError(f"invalid candidate scalar: {key}")
    missing = [key for key in CANDIDATE_FEATURE_FIELDS if key not in derived]
    if not missing:
        candidate_feature_vector(derived)
    return {"schema": list(CANDIDATE_FEATURE_FIELDS), "available": dict(derived),
            "missing": missing, "complete": not missing}


def assemble_input(context: np.ndarray, derived: Mapping[str, float]) -> np.ndarray:
    readiness = candidate_readiness(derived)
    if not readiness["complete"]:
        raise ValueError(f"planner features incomplete: {readiness['missing']}")
    return build_minimal_input(context, candidate_feature_vector(derived))
