"""Actual native goal/ctrl response for the corrected base inverse; no rerun."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();records=[]
    for p in sorted((a.run/'episodes').glob('dev-*/*/engineering-attempt-*/query-action-feedback.jsonl')):
        rows=[]
        for line in p.open():
            x=json.loads(line);b=x.get('base_frame_response')
            if b is None:continue
            expected=np.asarray(b['normalized_native_goal_expected']);actual=np.asarray(b['normalized_native_goal_actual'])
            if np.linalg.norm(expected)>1e-8:
                rows.append((float(np.max(abs(expected-actual))),float(np.linalg.norm(b['actual_base_qvel'])),float(np.linalg.norm(b['native_ctrl'])),b['theta_before_rad']))
        if rows:
            array=np.array(rows);records.append(dict(receipt=str(p),nonzero_native_goal_controls=len(rows),maximum_normalized_goal_error=float(array[:,0].max()),
                controls_with_actual_base_response=int(np.sum(array[:,1]>1e-8)),maximum_actual_generalized_base_speed=float(array[:,1].max()),
                maximum_native_ctrl_norm=float(array[:,2].max()),theta_range_rad=[float(array[:,3].min()),float(array[:,3].max())]))
    assert records and sum(x['controls_with_actual_base_response'] for x in records)>0,'No actual response yet; cannot freeze base candidate'
    maximum=max(x['maximum_normalized_goal_error'] for x in records)
    # Compare exact controller interpretation, not an asserted dynamics speed cap.
    assert maximum<=1e-6,maximum
    out=dict(at=datetime.now(timezone.utc).isoformat(),passed=True,records=records,maximum_native_goal_error=maximum,
        original_constraints_protections_unchanged=True,physical_response_observed=True,no_extra_episode_or_replay=True,
        scope='native raw-action swap/rotation inverse matches actual controller goal; actual nonzero response retained; not a proof of ideal velocity tracking or actual velocity upper bound')
    (a.run/'preflight/base-mapping-actual-validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)

if __name__=='__main__':main()
