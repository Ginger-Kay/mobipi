import h5py
import numpy as np
import pytest

from mobiwam.extract_features import VISUAL_KEYS
from mobiwam.reference_feature_interface import (
    PROPRIO_KEYS, assemble_input, candidate_readiness, source_observations,
)


def test_only_pre_action_frame_and_allowed_sensors(tmp_path):
    path = tmp_path / "rollout.hdf5"
    with h5py.File(path, "w") as f:
        obs = f.create_group("data/demo_0/obs")
        for key in VISUAL_KEYS:
            obs[key] = np.stack([np.full((4, 4, 3), 17, np.uint8),
                                 np.full((4, 4, 3), 250, np.uint8)])
        for key in PROPRIO_KEYS:
            obs[key] = np.array([[1., 2.], [np.nan, np.nan]])
        obs["object-state"] = np.full((2, 10), np.nan)
        obs["robot0_success"] = np.ones((2, 1))
    result = source_observations(path)
    assert set(result) == set((*VISUAL_KEYS, *PROPRIO_KEYS))
    assert result[VISUAL_KEYS[0]].shape == (1, 3, 4, 4)
    assert np.allclose(result[VISUAL_KEYS[0]], 17 / 255)
    assert result[PROPRIO_KEYS[0]].tolist() == [[1., 2.]]


def test_missing_fields_never_become_zero_padded_training_example():
    report = candidate_readiness({"route_E": 1.})
    assert not report["complete"]
    assert "minimum_continuous_clearance_m" in report["missing"]
    with pytest.raises(ValueError, match="incomplete"):
        assemble_input(np.ones((4, 1024)), report["available"])


@pytest.mark.parametrize("field", ["success", "outcome", "qpos", "stratum", "metadata"])
def test_outcome_and_raw_state_rejected(field):
    with pytest.raises(ValueError, match="only schema"):
        candidate_readiness({field: {"success": True}})


def test_nonfinite_rejected():
    with pytest.raises(ValueError, match="invalid"):
        candidate_readiness({"route_E": np.nan})
