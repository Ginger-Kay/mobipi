import copy
import pytest
from mobiwam.dr_v04_r2_summary import ADMITTED,ROUTES,evaluate,train_best_fixed,route_oracle

def dataset():
    roster=[];rows=[]
    for i in range(36):
        gid=ADMITTED if i==0 else f'g{i}'
        split='train' if i<24 else 'validation'
        roster.append(dict(group_id=gid,split=split,task='drawer' if i%2 else 'microwave',geometry_preoutcome_choice='E'))
        for route in ROUTES:
            success=(route=='E' and i not in (24,25)) or (route=='D' and i in (24,25))
            rows.append(dict(group_id=gid,split=split,route=route,success=success,failure=False,progress=float(success),base_path_m=1.,completion_time_s=2.,machine_eligible_for_gate=True))
    return rows,roster

def test_gate_and_exception_counts_do_not_authorize_training():
    rows,roster=dataset();out=evaluate(rows,roster,True)
    assert out['gate_status']=='mechanical_gate_pass_pending_research_review'
    assert out['main']['oracle_gain_vs_best_fixed']==2
    assert out['main']['oracle_gain_vs_geometry']==2
    assert out['main']['train_sources']==24
    assert out['excluding_seed109_sensitivity']['train_sources']==23
    assert not out['formal_train_ready'] and not out['training_authorized']

def test_complete_numbers_do_not_discharge_qualification():
    rows,roster=dataset();out=evaluate(rows,roster)
    assert out['main']['numerical_gate_pass'] and out['gate_status']=='gate_pending_review'

@pytest.mark.parametrize('key,value',[('failure',None),('success',None),('machine_eligible_for_gate',False)])
def test_missing_is_not_zero_or_dropped(key,value):
    rows,roster=dataset();rows[0][key]=value;out=evaluate(rows,roster,True)
    assert out['main'] is None and out['gate_status']=='gate_pending_review'
    assert out['expected_sources']==36 and out['expected_routes']==108

def test_duplicate_foreign_and_wrong_split_rejected():
    rows,roster=dataset()
    for change in ('duplicate','foreign','split'):
        bad=copy.deepcopy(rows)
        if change=='duplicate':bad[-1]=bad[0]
        elif change=='foreign':bad[0]['group_id']='sealed-test'
        else:bad[0]['split']='validation'
        with pytest.raises(ValueError):evaluate(bad,roster,True)

def test_validation_cannot_change_train_fixed():
    rows,roster=dataset();train={r['group_id'] for r in roster if r['split']=='train'}
    before=train_best_fixed(rows,train)
    for r in rows:
        if r['split']=='validation':r.update(success=r['route']=='A',progress=1.)
    assert before==train_best_fixed(rows,train)=='E'

def test_oracle_failure_then_progress_then_path_then_time_order():
    rows,_=dataset();rs=rows[:3]
    for r in rs:r.update(success=True,failure=False,progress=1.,base_path_m=1.,completion_time_s=2.)
    assert route_oracle(rs)=='E'
    rs[0]['failure']=True;assert route_oracle(rs)=='D'
    rs[2]['base_path_m']=.5;assert route_oracle(rs)=='A'
    rs[1]['progress']=1.1;assert route_oracle(rs)=='D'

def test_only_one_oracle_gain_fails_numeric_gate():
    rows,roster=dataset()
    for r in rows:
        if r['group_id']=='g24' and r['route']=='E':r.update(success=True,progress=1.)
    out=evaluate(rows,roster,True)
    assert out['main']['oracle_gain_vs_best_fixed']==1
    assert out['gate_status']=='mechanical_gate_fail_no_model_recommendation'

def test_exclusion_reselects_best_fixed_without_switching_main():
    rows,roster=dataset()
    for r in rows:
        if r['split']=='train':
            r['success']=r['route']=='D' or (r['route']=='E' and r['group_id']!=ADMITTED)
            r['progress']=float(r['success'])
    out=evaluate(rows,roster,True)
    assert out['main']['best_fixed']=='D'
    assert out['excluding_seed109_sensitivity']['best_fixed']=='E'
    assert out['sensitivity_best_fixed_changed']
