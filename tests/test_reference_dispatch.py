import pytest
from mobiwam.reference_dispatch import rejection_reason
from mobiwam.reference_dispatch import target_finger_contact


@pytest.mark.parametrize('fixture', ['stack_4_main_group_2', 'microwave_main_group',
                                   'island_cab_left_island_group_4'])
def test_contact_binding_allows_only_target_fingers(fixture):
    contact = dict(geom1=fixture+'_door_handle_handle',
                   geom2='gripper0_right_finger1_pad_collision')
    assert target_finger_contact(contact, fixture, handle_only=True)
    assert target_finger_contact(dict(geom1=contact['geom2'], geom2=contact['geom1']), fixture)
    assert not target_finger_contact(contact, 'unrelated_fixture')
    assert not target_finger_contact(dict(contact, geom1=fixture+'0_handle'), fixture)
    assert not target_finger_contact(dict(contact, geom2='gripper0_right_hand_collision'), fixture)
    assert not target_finger_contact(dict(contact, geom1=fixture+'_door'), fixture, handle_only=True)
    with pytest.raises(ValueError): target_finger_contact(contact, '')


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
