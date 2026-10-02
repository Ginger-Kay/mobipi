"""Mocked collectors exercise the real coordinator, with no GPU/physics/model."""
import hashlib,json,sys
import pytest
from pathlib import Path
import dr_v04_r2_parallel as coordinator


def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))


@pytest.mark.parametrize('failed_group',[None,'g5'])
def test_actual_coordinator_fifo_four_slots_and_unique_routes(tmp_path,monkeypatch,failed_group):
    order=[f'g{i}' for i in range(35)];rows=[dict(group_id=g,task='drawer' if i%2 else 'door',split='train',route_order=['D','E','A']) for i,g in enumerate(order)]
    binding=dict(run_id='run',parent_freeze=str(tmp_path/'freeze.json'),allowed_group_ids=order,scientific_rows=rows,parallel_gpu_indices=[0,1,2,3],new_route_budget=105,training_authorized=False,execution_code_commit='frozen')
    bp=tmp_path/'binding.json';save(bp,binding);permit=tmp_path/'permit.json';save(permit,dict(run_id='run',binding_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),requested_gpus=[0,1,2,3]))
    save(tmp_path/'preflight/first-batch-runtime-review.json',dict(runtime_checks_pass=True));save(tmp_path/'batches/checkpoint-04.json',dict(groups=order[:4]));(tmp_path/'videos').mkdir();(tmp_path/'audit').mkdir()
    monkeypatch.setattr(sys,'argv',['parallel','--run',str(tmp_path),'--binding',str(bp),'--manual-start-receipt',str(permit)])
    monkeypatch.setattr(coordinator,'completed_groups',lambda *args:order[:4].copy())
    monkeypatch.setattr(coordinator.subprocess,'check_output',lambda cmd,**kw:'frozen' if 'rev-parse' in cmd else '')
    monkeypatch.setattr(coordinator.time,'sleep',lambda n:None);monkeypatch.setenv('R2_ROBOCASA_PATH','unused')
    running={};dispatches=[];peak=[0]
    class FakeCollector:
        def __init__(self,cmd,stdout,stderr,env):
            self.group=cmd[cmd.index('--group-id')+1];out=Path(cmd[cmd.index('--output')+1]);gpu=int(env['CUDA_VISIBLE_DEVICES']);assert env['MUJOCO_EGL_DEVICE_ID']==str(gpu);assert gpu not in running.values()
            self.pid=1000+len(dispatches);self.polls=0;self.returncode=None;running[self.group]=gpu;dispatches.append(self.group);peak[0]=max(peak[0],len(running))
            save(out/'camera-preview/camera.json',dict(visibility_pass=True,source_integration_unchanged=True));save(out/'completed.json',dict(route_outcomes=3))
            for route in ('D','E','A'):
                a=out/'source'/route/'attempt-one';save(a/'task-video-manifest.json',dict(binding=dict(run_id='run')));(a/'original.mp4').write_bytes(b'fixture')
                rp=a/'replay/result.json';save(rp,dict(reproducible=True));save(out/f'route-{route}-replay.json',dict(reproducible=True,result=str(rp),sha256=hashlib.sha256(rp.read_bytes()).hexdigest()));save(out/f'route-{route}-dispatched.json',dict(path=str(a)))
        def poll(self):
            self.polls+=1
            if self.polls>=4 and self.returncode is None:self.returncode=int(self.group==failed_group);del running[self.group]
            return self.returncode
        def wait(self):
            while self.poll() is None:pass
            return self.returncode
    monkeypatch.setattr(coordinator.subprocess,'Popen',FakeCollector)
    def audit(root,freeze,group,route,attempt,out):
        return dict(group_id=group,route=route,attempt=str(attempt),replay_reproducible=True,original_video_sha256='fixture',completion_time_s=1,original_video_frames=1,checker_success=True,raw_executor_reason='fixture',machine_eligible_for_gate=True)
    monkeypatch.setattr(coordinator,'audit_one',audit)
    if failed_group:
        with pytest.raises(RuntimeError):coordinator.main()
        assert dispatches==order[4:4+len(dispatches)] and not running and peak[0]<=4
        assert json.loads((tmp_path/'status.json').read_text())['state']=='hold_engineering_diagnosis'
        return
    coordinator.main()
    assert dispatches==order[4:] and len(set(dispatches))==31 and peak[0]==4 and not running
    assert len(list((tmp_path/'batches').glob('group-*-dispatch.json')))==31
    assert len(list((tmp_path/'batches').glob('*/*/route-*-dispatched.json')))==93
    state=json.loads((tmp_path/'status.json').read_text());assert state['state']=='collection_and_machine_audits_complete' and state['training']==0 and state['sealed_test']==0
