import hashlib,json
import pytest
from mobiwam.reference_route_resume import completed_prefix

def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
def make(out,route):
    a=out/'source-s'/route/'attempt-one'
    save(a/'result.json',dict(route=route,steps=2))
    rp=a/'replay-one/result.json';save(rp,dict(reproducible=True,steps=2,max_state_abs_error=0))
    save(out/f'route-{route}-start.json',dict(route=route,run_id='run',group_id='group'))
    save(out/f'route-{route}-dispatched.json',dict(route=route,path=str(a)))
    save(out/f'route-{route}-replay.json',dict(result=str(rp),sha256=hashlib.sha256(rp.read_bytes()).hexdigest(),reproducible=True))

def test_completed_prefix_only(tmp_path):
    make(tmp_path,'D');make(tmp_path,'E')
    assert [x['route'] for x in completed_prefix(tmp_path,['D','E','A'],'run','group')]==['D','E']

def test_partial_route_refused(tmp_path):
    make(tmp_path,'D');(tmp_path/'route-D-replay.json').unlink()
    with pytest.raises(ValueError):completed_prefix(tmp_path,['D','E','A'],'run','group')

def test_gap_refused(tmp_path):
    make(tmp_path,'E')
    with pytest.raises(ValueError):completed_prefix(tmp_path,['D','E','A'],'run','group')

def test_unledgered_outcome_refused(tmp_path):
    (tmp_path/'source-s/D/attempt-one').mkdir(parents=True)
    with pytest.raises(ValueError):completed_prefix(tmp_path,['D','E','A'],'run','group')

def test_replay_tampering_refused(tmp_path):
    make(tmp_path,'D');save(tmp_path/'source-s/D/attempt-one/replay-one/result.json',{})
    with pytest.raises(ValueError):completed_prefix(tmp_path,['D','E','A'],'run','group')

def test_foreign_run_refused(tmp_path):
    make(tmp_path,'D')
    with pytest.raises(ValueError):completed_prefix(tmp_path,['D','E','A'],'different','group')
