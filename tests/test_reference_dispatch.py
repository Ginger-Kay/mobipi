import pytest
from mobiwam.reference_dispatch import rejection_reason


def test_strict_dispatch_rejects_missing_or_invalid_complete_path():
    assert rejection_reason({'D_prefix_executable': True}, 'D', True) == 'missing_or_ambiguous_full_plan'
    row = dict(route='A', hard_valid=False, features={'hard_valid': 0.}, collision={'valid': False})
    assert rejection_reason({'records': [row]}, 'A', True) == 'full_plan_not_hard_valid'


def test_strict_dispatch_accepts_only_consistent_valid_record():
    row = dict(route='A', hard_valid=True, features={'hard_valid': 1.}, collision={'valid': True}, source_qpos_unchanged=True)
    assert rejection_reason({'records': [row]}, 'A', True) is None
    row['collision']['valid'] = False
    assert rejection_reason({'records': [row]}, 'A', True) == 'full_plan_not_hard_valid'


def test_d_prefix_rejection_applies_in_both_modes():
    for strict in (False, True):
        assert rejection_reason({'D_prefix_executable': False}, 'D', strict) == 'no_ik_and_swept_prefix_valid_dock'
    assert rejection_reason({}, 'A', False) is None  # historical development mode
    with pytest.raises(ValueError): rejection_reason({}, 'unknown', True)
