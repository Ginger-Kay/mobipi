import numpy as np
from mobiwam.pi05_data_learning import labels,SCALES


def test_compute_truncation_preserves_observed_continuous_fields_without_failure_label():
    y,mask=labels(dict(status='compute-timeout',usable_scientific_outcome=True,native_success=False,
        terminal_native_opening=.8,actual_base_path_m=.12,terminal_duration_s=73.),dict(native_collision=False))
    assert not mask[0] and np.isnan(y[0])
    assert mask[2:].all() and np.allclose(y[2:],[.2,.12,73.])


def test_real_budget_failure_and120second_units():
    y,mask=labels(dict(status='budget',usable_scientific_outcome=True,native_success=False,
        terminal_native_opening=.8,actual_base_path_m=.12,terminal_duration_s=120.),dict(native_collision=False))
    assert mask.all() and y[0]==0
    assert SCALES[-1]==120.
