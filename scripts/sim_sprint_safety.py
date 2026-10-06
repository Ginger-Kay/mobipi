"""Actual saved native swept clearance, separate from task/video checks."""
import argparse,hashlib,json,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from mobiwam.reference_collision import SweptGeometry
from mobiwam.task_video_identity import source_model

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--receipt',type=Path,required=True);a=ap.parse_args();q=json.loads(a.receipt.read_text())
    attempt=Path(q['attempt']);out=attempt/'sprint-safety-audit.json';assert not out.exists();source=attempt.parents[1]
    sha=hashlib.sha256((source/'model.xml').read_bytes()).hexdigest();model=source_model(str(source/'model.xml'),sha)
    binding=json.loads((source/'target-binding.json').read_text());z=np.load(attempt/'formal-native-substeps.npz',allow_pickle=False)
    assert z['qpos'].shape==(len(z['phases'])+1,model.nq)
    start=time.monotonic();sweep=SweptGeometry(model,target_prefix=binding['fixture_name'],margin=.0005).path(z['qpos'],z['phases'])
    monitor=json.loads((attempt/'formal-native-substeps-receipt.json').read_text());margin=json.loads((attempt/'joint-margin-monitor.json').read_text())
    native_contact=monitor.get('forbidden_contact') is not None
    passed=bool(sweep['valid'] and not native_contact and margin['minimum'] and margin['minimum']['margin_rad']>.015)
    result=dict(created_at=datetime.now(timezone.utc).isoformat(),attempt=str(attempt),model_sha256=sha,
        actual_native_swept_geometry=sweep,clearance_pass=bool(sweep['valid']),native_collision=native_contact,
        joint_margin=margin,all_safety_pass=passed,native_success=bool(q['native_success']),safety_qualified_success=bool(passed and q['native_success'] and q.get('route_semantics_pass',True)),
        elapsed_seconds=time.monotonic()-start,full_replay_performed=False,formal_train_ready=False,
        evidence_scope='conservative recorded native interval0.5mm sweep plus native forbidden-contact/joint monitor; no formal Gate or human acceptance')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
