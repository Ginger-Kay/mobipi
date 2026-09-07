#!/usr/bin/env python3
"""Inventory V3 artifacts without treating geometry diagnostics as outcomes."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path

import numpy as np


def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def main(root):
    import torch,mujoco,scipy
    now=datetime.now(timezone.utc).isoformat()
    decisions={};probes=steps=0
    details={}
    for task in ('CloseDrawer','CloseSingleDoor'):
        for kind in ('probes','mapping-supplement','mapping-friction'):
            p=read(root/kind/task/'probe.json')
            probes+=len(p['rows']);steps+=sum(len(r['base']) for r in p['rows'])
        folder=root/'pose-compiler'/task/'hard-feasibility-repair'
        decisions[task]=read(folder/'decision.json')
        if decisions[task]['selected'] is not None:
            raise RuntimeError('eligible geometry exists: continue runtime work before negative delivery')
        rows=[]
        for path in sorted(folder.glob('*-validity.json')):
            r=read(path)
            rows.append({'candidate_id':r['candidate_id'],'hard_valid':r['hard_valid'],
                         'source_ik':r['source_ik'],
                         'routes':{k:{'passed':v['passed'],'knots_attempted':len(v['knots']),'last_knot':v['knots'][-1]} for k,v in r['routes'].items()}})
        details[task]=rows
    write(root/'geometry-failure-summary.json',{'at':now,'decisions':decisions,'candidates':details,
          'interpretation':'bounded five frozen proposals per task; local IK failure is not proof of global task infeasibility'})
    inherited=[]
    parent=Path('/share/jhk/MobiWAM/artifacts/MMWAM-OBC-002/v2-1-articulation-a-video-cont-v1.0/20260907T023700Z-v2-1-articulation-a-video-cont-v1.0/qualification')
    for task in ('CloseDrawer','CloseSingleDoor'):
        record=read(parent/task/'a-v4-record.json');video=Path(record['video_path'])
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height,r_frame_rate,nb_frames:format=duration','-of','json',str(video)],text=True))
        inherited.append({'task':task,'path':str(video),'record':str(parent/task/'a-v4-record.json'),
                          'bytes':video.stat().st_size,'sha256':sha(video),'ffprobe':info,
                          'attribution':'V2.1 inherited diagnostic raw A, not a V3 route or eligible video'})
    write(root/'video-inventory.json',{'at':now,'v3_raw_videos':0,'v3_final_comparison_videos':0,
          'inherited_diagnostic_raw':inherited,'visual_gate':'not_run_no_eligible_routes',
          'static_geometry_frames':'geometry-frames/inventory.json'})
    counters={'probe_episodes':probes,'probe_env_steps':steps,'development_outcomes':0,'qualified_A':0,
              'final_comparison_records':0,'final_comparison_videos':0,'pose_proposals':10,
              'geometry_compiler_passes_per_task':3,'training_runs':0,'validation_reads':0,'test_reads':0}
    manifest={'run_id':root.name,'created_at':read(root/'continuity-receipt.json')['created_at'],'updated_at':now,
              'started_at':'2026-09-07T06:50:34Z','ended_at':now,'status':'bounded_source_geometry_hold',
              'review_status':'pending','researcher_video_approved':False,'claim_support':'none',
              'control_input_commit':'22b6d3540b035d8694b6a45a05231b45b6b3f738',
              'code_commit':subprocess.check_output(['git','-C','/share/jhk/MobiWAM/Mobipi','rev-parse','HEAD'],text=True).strip(),
              'geometry_final_execution_commit':'a947cc9','counters':counters,'decisions':decisions,
              'environment':{'python':sys.executable,'python_version':platform.python_version(),'torch':torch.__version__,
                             'torch_cuda':torch.version.cuda,'numpy':np.__version__,'mujoco':mujoco.__version__,'scipy':scipy.__version__},
              'launch':'launch-receipt.json','commands':['run-probes.sh','run-friction.sh','run-control-audit.sh','run-construct.sh'],
              'source_exclusion':'development-exclusion.json','profile':'movement-profile-v1.json',
              'notes':['profile does not establish visual distinguishability','mapping implementation is not runtime-qualified',
                       'original-source whole-robot contacts were discovered after the independent probes; no qualification used those states',
                       'task_success_reads in probes means no explicit checker queries by the probe; simulator-internal reward/checker calls are not counted',
                       'first compiler passes are superseded by the final clean-commit hard-feasibility pass; no new proposals were created',
                       'final videos and timed video self-check unavailable because no route became eligible']}
    write(root/'run-manifest.json',manifest)
    files=[root/'movement-profile-v1.json',root/'geometry-failure-summary.json',root/'video-inventory.json',root/'run-manifest.json']
    files.extend(sorted((root/'geometry-frames').glob('*')))
    write(root/'artifact-inventory.json',{'at':now,'files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files if p.is_file()]})
    print(json.dumps(counters),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
