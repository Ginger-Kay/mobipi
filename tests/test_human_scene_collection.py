import csv
import pytest
from mobiwam.human_scene_collection import validate_start,append_index,load_index,phase_for


def test_primary_failure_is_not_replaceable():
    cfg=dict(primary_enabled=True,frozen_at='2026-10-03T00:00:00+00:00',frozen_source='/source-1',route_order=['E','D','A'])
    rows=[dict(source_id='source-1',record_type='primary',route='E',machine_success=False)]
    with pytest.raises(ValueError,match='cannot be replaced'):validate_start(cfg,rows,'E','primary','jhk','/source-1')
    validate_start(cfg,rows,'D','primary','jhk','/source-1')
    with pytest.raises(ValueError,match='Source differs'):validate_start(cfg,rows,'D','primary','jhk','/source-2')


def test_pilot_cannot_silently_become_primary_or_engineering():
    validate_start({},[],'A','practice','jhk','/source-1')
    with pytest.raises(ValueError,match='not frozen'):validate_start({},[],'A','primary','jhk','/source-1')
    with pytest.raises(ValueError,match='Operator|operator'):validate_start({},[],'A','practice','','/source-1')
    with pytest.raises(ValueError,match='not enabled'):validate_start({},[],'E','engineering_record_smoke','jhk','/source-1')


def test_supplements_are_two_total_and_after_trio():
    cfg=dict(primary_enabled=True,frozen_at='yes',frozen_source='/s',route_order=['E','D','A'])
    rows=[dict(source_id='s',record_type='primary',route=x) for x in 'EDA']
    with pytest.raises(ValueError,match='Finish'):validate_start(cfg,rows[:2],'A','reference_supplement','jhk','/s')
    validate_start(cfg,rows,'A','reference_supplement','jhk','/s')
    rows += [dict(source_id='s',record_type='reference_supplement',route=x) for x in 'AE']
    with pytest.raises(ValueError,match='Two total'):validate_start(cfg,rows,'D','reference_supplement','jhk','/s')


def test_append_preserves_failure_and_refuses_duplicate(tmp_path):
    p=tmp_path/'index.csv'
    append_index(p,dict(attempt_id='a1',machine_success=False,stop_reason='human_stop'))
    append_index(p,dict(attempt_id='a2',machine_success=True,stop_reason='checker_success_10_steps'))
    with pytest.raises(ValueError,match='already indexed'):append_index(p,dict(attempt_id='a1'))
    rows=load_index(p);assert len(rows)==2 and rows[0]['machine_success']=='False'
    assert phase_for('D',False)=='navigate' and phase_for('D',True)=='manipulate'
