import json
import numpy as np
import pytest
from mobiwam.human_postcapture import claim, read_json, numeric_alignment, pair_records, summarize_sweeps

def test_claim_once(tmp_path):
    claim(tmp_path, tmp_path/'attempt', 'replay', {})
    with pytest.raises(ValueError):claim(tmp_path, tmp_path/'attempt', 'replay', {})

def test_json_corruption(tmp_path):
    p=tmp_path/'bad';p.write_text('{')
    with pytest.raises(json.JSONDecodeError):read_json(p,delay=0)

@pytest.mark.parametrize('nq,nv',[(83,79),(70,66)])
def test_dynamic_dimensions(nq,nv):
    rows=[dict(before=dict(qpos=[0]*nq,qvel=[0]*nv),after=dict(qpos=[0]*nq,qvel=[0]*nv))]
    states=np.zeros((2,1+nq+nv));states[1,0]=.05
    native=dict(qpos=np.zeros((26,nq)),sim_time=np.arange(26)*.002,step_index=np.zeros(25,int))
    assert numeric_alignment(np.zeros((1,12)),states,rows,native)['nq']==nq
    states[1,1]=1
    with pytest.raises(ValueError):numeric_alignment(np.zeros((1,12)),states,rows,native)

def test_pairs_keep_failures(tmp_path):
    p=tmp_path/'s.npy';np.save(p,[1,2])
    rows=[dict(record_type='primary',source='s',config_version='v',route=r,initial_integration=str(p),checker_success=r!='A') for r in 'EDA']
    assert pair_records(rows)[0]['failure_routes']==['A']
    assert not pair_records(rows[:2])[0]['paired_attempt_set_complete']
    with pytest.raises(ValueError):pair_records(rows+[rows[0]])
    q=tmp_path/'q.npy';np.save(q,[1,3]);rows[1]['initial_integration']=str(q)
    with pytest.raises(ValueError):pair_records(rows)

def test_sweep_partition():
    def part(a,b):return dict(begin_interval=a,end_interval=b,total_native_intervals=10,required_clearance_m=.0005,contact_rule_version='DR-v0.4-R3-exact-finger-pad',valid=True,checked_native_intervals=b,lower_bound_m=.01)
    assert summarize_sweeps([part(0,4),part(4,10)],10)['valid']
    with pytest.raises(ValueError):summarize_sweeps([part(0,4),part(5,10)],10)
    a=part(0,10);a['checked_native_intervals']=8
    with pytest.raises(ValueError):summarize_sweeps([a],10)
    a['valid']=False
    assert not summarize_sweeps([a],10)['valid']

def test_stage_duplicate_and_failure(tmp_path):
    import sys
    from human_postcapture_pipeline import run_stage
    out=tmp_path/'out';out.mkdir()
    args=(tmp_path/'attempt','replay',tmp_path/'ledger',{},[('probe',[sys.executable,'-c','raise SystemExit(1)'])],out,1)
    with pytest.raises(RuntimeError):run_stage(*args)
    with pytest.raises(ValueError):run_stage(*args)
    receipts=list((tmp_path/'ledger').glob('*/*.json'))
    assert len(receipts)==1 and read_json(receipts[0])['status']=='engineering_hold'

def test_live_json_transient(monkeypatch,tmp_path):
    original=__import__('pathlib').Path.read_text
    p=tmp_path/'live';p.write_text('{"ok":true}')
    calls=[]
    def read(self,*a,**kw):
        calls.append(1)
        return '{' if len(calls)==1 else original(self,*a,**kw)
    monkeypatch.setattr(type(p),'read_text',read)
    assert read_json(p,delay=0)=={'ok':True}
    assert len(calls)==2
