"""Nonlearning R2 counts and preregistered comparisons. Never fits a model."""
from collections import Counter
import math

ROUTES=('E','D','A')
ADMITTED='CloseDrawer-layout1-style0-seed109'

def values(row):
    keys=('success','failure','progress','base_path_m','completion_time_s')
    if any(row.get(k) is None for k in keys):raise ValueError('missing label is not zero')
    if type(row['success']) is not bool or type(row['failure']) is not bool:raise ValueError('boolean labels required')
    if any(not math.isfinite(float(row[k])) for k in keys):raise ValueError('nonfinite label')
    return (-int(row['success']),int(row['failure']),-float(row['progress']),float(row['base_path_m']),float(row['completion_time_s']))

def route_oracle(rows):
    if {r['route'] for r in rows}!=set(ROUTES) or len(rows)!=3:raise ValueError('incomplete oracle denominator')
    return min(rows,key=lambda r:(*values(r),ROUTES.index(r['route'])))['route']

def train_best_fixed(rows,group_ids):
    selected=[]
    for route in ROUTES:
        rs=[r for r in rows if r['group_id'] in group_ids and r['route']==route]
        if len(rs)!=len(group_ids) or {r['group_id'] for r in rs}!=set(group_ids):raise ValueError('incomplete fixed denominator')
        vals=[values(r) for r in rs]
        sums=tuple(sum(v[k] for v in vals) for k in range(5))
        selected.append(((*sums,ROUTES.index(route)),route))
    return min(selected)[1]

def metrics(rows):
    if not rows:raise ValueError('empty evaluation')
    for row in rows:values(row)
    n=len(rows)
    return dict(sources=n,successes=sum(r['success'] for r in rows),failures=sum(r['failure'] for r in rows),
                progress_mean=sum(r['progress'] for r in rows)/n,base_path_mean_m=sum(r['base_path_m'] for r in rows)/n,
                completion_time_mean_s=sum(r['completion_time_s'] for r in rows)/n)

def comparisons(rows,roster,train_ids):
    val=[g for g in roster if g['split']=='validation'];val_ids=[g['group_id'] for g in val]
    fixed=train_best_fixed(rows,train_ids);by={(r['group_id'],r['route']):r for r in rows}
    oracle={g:route_oracle([by[g,r] for r in ROUTES]) for g in val_ids}
    geometry={g['group_id']:g['geometry_preoutcome_choice'] for g in val}
    selections={**{'fixed_'+r:{g:r for g in val_ids} for r in ROUTES},'train_best_fixed':{g:fixed for g in val_ids},'geometry':geometry,'oracle':oracle}
    scores={k:metrics([by[g,r] for g,r in choices.items()]) for k,choices in selections.items()}
    by_task={t:{k:metrics([by[g,r] for g,r in choices.items() if next(x for x in val if x['group_id']==g)['task']==t])
                    for k,choices in selections.items()} for t in sorted({x['task'] for x in val})}
    gain_fixed=scores['oracle']['successes']-scores['train_best_fixed']['successes']
    gain_geometry=scores['oracle']['successes']-scores['geometry']['successes']
    conditions=dict(at_least_two_realized_best_routes=len(set(oracle.values()))>=2,oracle_gain_vs_best_fixed_at_least_two=gain_fixed>=2,
                    oracle_gain_vs_geometry_at_least_two=gain_geometry>=2)
    return dict(train_sources=len(train_ids),validation_sources=len(val_ids),best_fixed=fixed,validation=scores,
                validation_by_task=by_task,realized_best=oracle,realized_best_counts=dict(Counter(oracle.values())),
                geometry_preoutcome=geometry,oracle_gain_vs_best_fixed=gain_fixed,oracle_gain_vs_geometry=gain_geometry,
                numerical_gate_conditions=conditions,numerical_gate_pass=all(conditions.values()))

def evaluate(rows,roster,qualification_closed=False):
    if len(roster)!=36 or len({g['group_id'] for g in roster})!=36:raise ValueError('R2 requires exact 36-source roster')
    train={g['group_id'] for g in roster if g['split']=='train'};val={g['group_id'] for g in roster if g['split']=='validation'}
    if len(train)!=24 or len(val)!=12 or ADMITTED not in train:raise ValueError('24/12 frozen split and admitted group required')
    expected={(g['group_id'],r) for g in roster for r in ROUTES};keys=[(r['group_id'],r['route']) for r in rows]
    if len(set(keys))!=len(keys) or set(keys)!=expected:raise ValueError('duplicate/missing/foreign row denominator')
    for row in rows:
        expected_split='train' if row['group_id'] in train else 'validation'
        if row['split']!=expected_split:raise ValueError('split mismatch')
    complete=all(r.get('machine_eligible_for_gate') is True and all(r.get(k) is not None for k in ('success','failure','progress','base_path_m','completion_time_s')) for r in rows)
    out=dict(expected_sources=36,expected_routes=108,expected_train_sources=24,expected_validation_sources=12,
             machine_eligible_routes=sum(r.get('machine_eligible_for_gate') is True for r in rows),
             complete_machine_dataset=complete,formal_train_ready=False,training_authorized=False,cv_fits=0,test_routes=0,
             gate_status='gate_pending_review',main=None,excluding_seed109_sensitivity=None)
    if not complete:return out
    main=comparisons(rows,roster,train);sensitive=comparisons(rows,roster,train-{ADMITTED})
    out.update(main=main,excluding_seed109_sensitivity=sensitive,
               sensitivity_best_fixed_changed=main['best_fixed']!=sensitive['best_fixed'],
               sensitivity_numerical_gate_changed=main['numerical_gate_pass']!=sensitive['numerical_gate_pass'],
               nonnumerical_qualification_closed=qualification_closed)
    if qualification_closed:
        out['gate_status']='mechanical_gate_pass_pending_research_review' if main['numerical_gate_pass'] else 'mechanical_gate_fail_no_model_recommendation'
    return out
