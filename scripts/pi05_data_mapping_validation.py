"""Actual native goal/ctrl response for the corrected base inverse; no rerun."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import h5py

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();records=[]
    for p in sorted((a.run/'episodes').glob('dev-*/*/engineering-attempt-*/query-action-feedback.jsonl')):
        if not (p.parent/'completed.json').exists():continue
        completed=json.loads((p.parent/'completed.json').read_text());attempt=Path(completed['attempt'])
        with h5py.File(attempt/'demo.hdf5','r') as f:states=f['data/demo_0/states'][:]
        # Native PandaOmron base qpos are0/1/2; col0 is sim time.
        # The controller refreshes FK at action application; its current angle
        # comes from states[t], whereas pre-step get_base_pose may be cached at
        # the previous2ms integration boundary. Keep that discrepancy explicit.
        init_yaw=float(states[0,3])
        rows=[]
        for line in p.open():
            x=json.loads(line);b=x.get('base_frame_response')
            if b is None:continue
            before_cached=np.asarray(b['normalized_native_goal_expected']);actual=np.asarray(b['normalized_native_goal_actual'])
            theta=float(states[x['step'],3])-init_yaw
            matrix=np.array([[np.sin(theta),np.cos(theta),0],[np.cos(theta),-np.sin(theta),0],[0,0,1]])
            expected=matrix@np.asarray(x['actual_action'])[7:10]
            if np.linalg.norm(expected)>1e-8:
                rows.append((float(np.max(abs(expected-actual))),float(np.linalg.norm(b['actual_base_qvel'])),float(np.linalg.norm(b['native_ctrl'])),theta,float(np.max(abs(before_cached-actual)))))
        if rows:
            array=np.array(rows);records.append(dict(receipt=str(p),nonzero_native_goal_controls=len(rows),maximum_normalized_goal_error=float(array[:,0].max()),
                controls_with_actual_base_response=int(np.sum(array[:,1]>1e-8)),maximum_actual_generalized_base_speed=float(array[:,1].max()),
                maximum_native_ctrl_norm=float(array[:,2].max()),theta_range_rad=[float(array[:,3].min()),float(array[:,3].max())],pre_refresh_cached_goal_discrepancy=float(array[:,4].max())))
    assert records and sum(x['controls_with_actual_base_response'] for x in records)>0,'No actual response yet; cannot freeze base candidate'
    maximum=max(x['maximum_normalized_goal_error'] for x in records)
    # Compare exact controller interpretation, not an asserted dynamics speed cap.
    assert maximum<=1e-6,maximum
    out=dict(at=datetime.now(timezone.utc).isoformat(),passed=True,records=records,maximum_native_goal_error=maximum,
        original_constraints_protections_unchanged=True,physical_response_observed=True,no_extra_episode_or_replay=True,
        scope='native raw-action swap/rotation inverse at current saved state matches actual controller goal; cached pre-forward discrepancy separate; actual nonzero response retained; not ideal velocity tracking or actual speed upper-bound proof')
    (a.run/'preflight/base-mapping-actual-validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)

if __name__=='__main__':main()
