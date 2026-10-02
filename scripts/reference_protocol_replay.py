"""Real saved-action replay with shared events, field receipts and no state injection."""
import argparse,json
from pathlib import Path
import numpy as np,h5py,imageio.v2 as imageio
from teleop_reference import Reference,write_json,stamp
from mobiwam.reference_controller_events import load_replay_events,apply_event,snapshot
from mobiwam.replay_diagnostics import state_fields,summarize_drift

def replay_attempt(ref,attempt,output,plan=None,video=False):
    attempt=Path(attempt).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    result=json.loads((attempt/'result.json').read_text());events,lineage=load_replay_events(attempt,result,plan)
    with h5py.File(attempt/'demo.hdf5') as h:
        actions=h['data/demo_0/actions'][:];states=h['data/demo_0/states'][:]
    if len(states)!=len(actions)+1 or not np.isfinite(states).all() or not np.isfinite(actions).all():raise ValueError('invalid native state/actions')
    ref.source=attempt.parent.parent;ref.route=result['route'];ref.restore();initial=ref.env.sim.get_state().flatten().copy()
    initial_error=float(np.max(np.abs(initial-states[0])))
    if initial_error>1e-10:raise ValueError('initial Source restore differs')
    applied=[];event_receipts=[];actual=[initial];errors=[];controllers=[]
    trace=[json.loads(x) for x in (attempt/'trace.jsonl').read_text().splitlines()]
    writer=imageio.get_writer(output/'replay.mp4',fps=20,codec='libx264',quality=7,macro_block_size=None) if video else None
    try:
        for i,action in enumerate(actions):
            for event in events:
                if event['step']==i:
                    event_receipts.append(dict(step=i,receipt=apply_event(ref,event)));applied.append((event['event'],i))
            # No recorded state write inside this loop: only original action + events.
            ref.env.step(action)
            value=ref.env.sim.get_state().flatten().copy();actual.append(value);errors.append(float(np.max(np.abs(value-states[i+1]))))
            if any(abs(i-e['step'])<=2 for e in events) or i==0 or i==len(actions)-1:controllers.append(dict(after_action=i,controller=snapshot(ref)))
            if writer:writer.append_data(ref.frame(trace[i]['camera']))
            if i%100==0:print('REAL_REPLAY',attempt.parent.name,i,'state_error',errors[-1],flush=True)
    finally:
        if writer:writer.close()
    from mobiwam.reference_terminal_step import terminal_record,replay_terminal
    feedback=[json.loads(x) for x in (attempt/'feedback.jsonl').read_text().splitlines()]
    partial=terminal_record(attempt,result,feedback);terminal=None
    if partial is not None:
        for event in events:
            if event['step']==len(actions):
                event_receipts.append(dict(step=len(actions),receipt=apply_event(ref,event)));applied.append((event['event'],len(actions)))
        terminal=replay_terminal(ref,partial)
        write_json(output/'terminal-step.json',terminal)
    if len(applied)!=len(events):raise ValueError('missing event application')
    model,_=ref.model_data();actual=np.asarray(actual);diff=np.abs(actual-states)
    np.savez_compressed(output/'replayed-states-and-field-errors.npz',actual=actual,absolute_field_errors=diff)
    fields=summarize_drift(states,actual,state_fields(model),threshold=1e-5);write_json(output/'field-errors.json',fields)
    write_json(output/'controller-event-receipts.json',dict(events=event_receipts,controller_windows=controllers,legacy_reconstruction=lineage,event_application_timing='before saved zero-based action; no observation/control-policy regeneration',unknown_event_fail_closed=True))
    first=next((i for i,e in enumerate(errors) if e>1e-5),None)
    out=dict(ended_at=stamp(),attempt=str(attempt),steps=len(actions),route=result['route'],initial_error=initial_error,
             max_state_abs_error=max(errors,default=0.),first_state_error_gt_1e_5=first,
             state_errors=errors,checker_success=bool(ref.env._check_success()),expected_checker_success=result['checker_success'],
             reproducible=(terminal is None or terminal['reproducible']) and max(errors,default=0.)<=1e-5 and bool(ref.env._check_success())==bool(result['checker_success']),
             replay_kind='real native saved actions + versioned controller events; no per-step state injection',applied_events=applied,
             original_events_preserved=True,video_generated=video,scientific_route_outcomes=0,formal_train_ready=False)
    out['first_state_error_gt_1e-5']=first
    out['terminal_step']=terminal
    write_json(output/'result.json',out);return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--output',required=True);p.add_argument('--plan');p.add_argument('--video',action='store_true');a=p.parse_args()
    attempt=Path(a.attempt).resolve();out=Path(a.output).resolve()
    ref=Reference(argparse.Namespace(output=str(out.parent/('environment-'+out.name)),task='CloseDrawer',layout=0,style=0,seed=7,self_test=True,source=str(attempt.parent.parent),replay_attempt=None,resume_attempt=None,width=1920,height=1080))
    try:
        result=replay_attempt(ref,attempt,out,a.plan,a.video);print(json.dumps({k:v for k,v in result.items() if k!='state_errors'}))
    finally:
        if ref.observation_renderer:ref.observation_renderer.close()
        if ref.renderer:ref.renderer.close()
        ref.env.close()
if __name__=='__main__':main()
