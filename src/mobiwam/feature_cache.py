from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def feature_cache_key(
    *, source_checksum: str, candidate_checksum: str, encoder_revision: str
) -> str:
    if not all((source_checksum, candidate_checksum, encoder_revision)):
        raise ValueError("all cache-key components must be non-empty")
    material = f"{source_checksum}\0{candidate_checksum}\0{encoder_revision}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FeatureCacheRecord:
    key: str
    path: Path
    shape: tuple[int, ...]
    dtype: str


class FeatureCache:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("cache key must be a lowercase SHA-256 digest")
        return self.root / key[:2] / f"{key}.npy"

    def put(self, key: str, value: np.ndarray) -> FeatureCacheRecord:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.partial-{os.getpid()}")
        with partial.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
        return FeatureCacheRecord(key, path, tuple(value.shape), str(value.dtype))

    def get(self, key: str) -> np.ndarray:
        path = self.path_for(key)
        return np.load(path, allow_pickle=False)
