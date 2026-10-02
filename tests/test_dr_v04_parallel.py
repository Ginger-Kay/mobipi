import pytest
from mobiwam.dr_v04_parallel import choose_gpu,batches,group_output,completed_groups
from mobiwam.dr_v04_gpu_inventory import require_occupancy_only


def test_fifo_batch_budget_and_original_paths(tmp_path):
    order=list(range(35));chunks=batches(order)
    assert [g for chunk in chunks for g in chunk]==order
    assert [len(c) for c in chunks]==[4,6,6,6,6,6,1]
    assert str(group_output(tmp_path,5,'g')).endswith('batches/01/05-g')
    assert str(group_output(tmp_path,11,'g')).endswith('batches/02/11-g')


def test_no_slot_no_dispatch_and_outcome_blind_task_balance():
    with pytest.raises(ValueError):choose_gpu(set(),'drawer',{})
    counts={(0,'drawer'):2,(1,'door'):2}
    assert choose_gpu({0,1},'drawer',counts)==1
    assert choose_gpu({0,1},'door',counts)==0
    assert choose_gpu({3},'drawer',counts)==3


def test_partial_group_cannot_be_recollected(tmp_path):
    (tmp_path/'batches/00/01-g').mkdir(parents=True)
    with pytest.raises(ValueError):completed_groups(tmp_path,['g'],{'g':{'route_order':['E','D','A']}},'run')


def test_unknown_gpu_context_and_allocation_changes_refused():
    rows=[dict(index=i,uuid=str(i),compute=[dict(pid=i+100)],graphics=[]) for i in range(4)]
    expected={str(i):dict(uuid=str(i),occupancy_pid=i+100) for i in range(4)}
    require_occupancy_only(rows,expected)
    rows[1]['graphics']=[dict(pid=999)]
    with pytest.raises(ValueError):require_occupancy_only(rows,expected)
    rows[1]['graphics']=[];rows[2]['compute'].append(dict(pid=998))
    with pytest.raises(ValueError):require_occupancy_only(rows,expected)
    with pytest.raises(ValueError):require_occupancy_only(rows[:3],expected)
