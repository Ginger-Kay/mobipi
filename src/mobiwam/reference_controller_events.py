"""Versioned, shared control-state transitions. No simulator state injection."""
import copy,hashlib,json
from pathlib import Path
import numpy as np

SCHEMA='reference-controller-events-v1'
EVENTS={'initial_stow_nullspace','dock_settled_reobserve_feedback_reset'}

def snapshot(ref):
    arm=ref.robot.part_controllers['right'];fields={}
    for key in ('initial_joint','goal_pos','goal_ori','origin_pos','origin_ori','goal_update_mode','_goal_update_mode','input_ref_frame'):
        if hasattr(arm,key):
            value=getattr(arm,key)
            if isinstance(value,np.ndarray):value=value.tolist()
            if isinstance(value,(float,int,str,bool,list,type(None))):fields[key]=value
    return dict(arm=fields,rng=copy.deepcopy(ref.env.rng.bit_generator.state),
                policy_kind='stateless reference-feedback; no learned policy history')

def apply_event(ref,event):
    if event.get('event') not in EVENTS:raise ValueError('unknown controller event')
    if event.get('timing')!='before_action':raise ValueError('controller event timing must be before_action')
    goal=np.asarray(event.get('arm_nullspace_goal'),float)
    if goal.shape!=(7,) or not np.isfinite(goal).all():raise ValueError('missing or invalid nullspace event payload')
    before=snapshot(ref)
    if event['event']=='dock_settled_reobserve_feedback_reset':
        ref.env._get_observations(force_update=True)
        ref.robot.composite_controller.update_state()
        ref.robot.part_controllers['right'].set_goal_update_mode('achieved')
        ref.robot.part_controllers['right'].set_goal(np.zeros(6))
    ref.robot.part_controllers['right'].initial_joint=goal.copy()
    return dict(event=event,before=before,after=snapshot(ref))

def emit_event(ref,event):
    receipt=apply_event(ref,event)
    if not ref.recording:raise ValueError('controller event emitted outside active recording')
    ref.recording['events'].append(dict(event,controller_before=receipt['before'],controller_after=receipt['after']))
    return receipt

def validate_events(events,route,n,required_stow=False):
    seen=set();last=-1
    for event in events:
        kind=event.get('event');step=event.get('step')
        if kind not in EVENTS:raise ValueError('unknown controller event')
        if route!='D':raise ValueError('E/A cannot carry D reset events')
        if type(step) is not int or not 0<=step<n or step<last:raise ValueError('controller event step/order differs')
        if (kind,step) in seen or any(k==kind for k,s in seen):raise ValueError('duplicate controller event')
        if kind=='initial_stow_nullspace' and step!=0:raise ValueError('initial stow must precede action0')
        if event.get('timing')!='before_action':raise ValueError('event timing missing or incorrect')
        goal=np.asarray(event.get('arm_nullspace_goal'),float)
        if goal.shape!=(7,) or not np.isfinite(goal).all():raise ValueError('missing controller event payload')
        seen.add((kind,step));last=step
    if required_stow and not any(e['event']=='initial_stow_nullspace' for e in events):raise ValueError('missing initial stow event')
    return events

def active_path(value):
    text=str(value)
    for prefix in ('/share/jhk/MobiWAM/','/share/personal/haokaijiang/MobiWAM/'):
        if text.startswith(prefix):text='/share/personal/chensiyu/haokaijiang/MobiWAM/'+text[len(prefix):];break
    return Path(text)

def load_replay_events(attempt,result,explicit_plan=None):
    """Legacy events reconstructed only from original code and sealed plan.

    Legacy dock reset occurs after action was computed, but before env.step.
    Replay uses saved actions, so applies exactly before the same action index.
    """
    attempt=Path(attempt);n=result['steps'];events=copy.deepcopy(result.get('events',[]));lineage=[]
    if result.get('controller_events_schema')==SCHEMA:
        validate_events(events,result['route'],n,result.get('initial_stow_required',False));return events,lineage
    if any(e.get('event')!='dock_settled_reobserve_feedback_reset' for e in events):raise ValueError('unknown legacy controller event')
    for event in events:
        event['timing']='before_action';lineage.append(dict(event=event['event'],step=event['step'],basis='original reference_executor.run_route reset immediately before ref.step; action already computed',reconstruction='timing only; original nullspace payload preserved'))
    if result['route']=='D':
        plan=active_path(explicit_plan) if explicit_plan else None
        if plan is None:
            candidates=(attempt.parents[2]/'plan-reuse-receipt.json',attempt.parents[2]/'formal-preexecution-receipt.json')
            if candidates[0].is_file():plan=active_path(json.loads(candidates[0].read_text())['plan_run'])
            elif candidates[1].is_file():
                receipt=json.loads(candidates[1].read_text());freeze=json.loads(active_path(receipt['preoutcome_freeze']).read_text());row=next(x for x in freeze['primary'] if x['group_id']==receipt['group_id']);plan=active_path(row['plan_run'])
        if plan is None:raise ValueError('legacy D requires exact sealed planning provenance')
        dock_path=plan/'dock-plan.json';dock=json.loads(dock_path.read_text());stow=dock['selected'].get('stow_target')
        if stow is not None:
            from reference_stow import load_stow
            # Serialized target includes arm_qpos; load_stow also resolves the geometric pose.
            goal=np.asarray(stow['arm_qpos'],float)
            event=dict(event='initial_stow_nullspace',step=0,timing='before_action',arm_nullspace_goal=goal.tolist())
            events.insert(0,event);lineage.append(dict(event=event,basis='original run_route sets initial_joint from dock.selected.stow_target before action0',plan=str(plan),dock_sha256=hashlib.sha256(dock_path.read_bytes()).hexdigest()))
        lineage.append(dict(legacy_plan=str(plan),initial_stow_required=stow is not None,existing_original_events_untouched=True))
    validate_events(events,result['route'],n,any(e['event']=='initial_stow_nullspace' for e in events));return events,lineage
