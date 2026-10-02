import pytest
import dr_v04_r2_monitor as monitor

@pytest.mark.parametrize('bad', ['origin','branch','dirty','diverged'])
def test_publication_rejects_unknown_repo_changes_before_writing(tmp_path,monkeypatch,bad):
    run=tmp_path/'project/artifacts/exp/v04/r2/run';run.mkdir(parents=True)
    calls=[]
    def git(repo,*args):
        calls.append(args)
        if args[:2]==('remote','get-url'):return 'git@github-jhk:someone/other.git' if bad=='origin' else 'git@github-jhk:Ginger-Kay/MM-WAM-Research.git'
        if args[0]=='branch':return 'other' if bad=='branch' else 'main'
        if args[0]=='status':return ' M unknown.md' if bad=='dirty' else ''
        if args[0]=='fetch':return ''
        if args[0]=='rev-list':return '0 1' if bad=='diverged' else '0 0'
        raise AssertionError('unexpected mutating Git action '+repr(args))
    monkeypatch.setattr(monitor,'git',git)
    with pytest.raises(ValueError):monitor.publish(run,tmp_path/'nonexistent-summary','checkpoint-10',False)
    assert not (tmp_path/'project/control').exists()
    assert not any(x[0] in ('add','commit','push','reset','clean','stash','checkout') for x in calls)


def test_latest_view_updates_owner_without_rewriting_historical_counts():
    original='# Initial\nupdated_at: old\nstatus: running\ncount 1\n'
    first=monitor.current_document(original,'count 12','running','Compute Execution','new')
    second=monitor.current_document(first,'count 105','complete','Research Control','final')
    current,history=second.split('<!-- R2_CHECKPOINT_HISTORY -->',1)
    assert 'status: `complete`' in current and 'control_write_owner: `Research Control`' in current
    assert 'count 105' in current and 'count 12' not in current
    assert original in history and 'count 12' in history
