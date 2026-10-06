"""Freeze minimal roster and measured pre-outcome inputs; no policy/outcome call."""
import argparse,hashlib,json,re
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from teleop_reference import Reference,write_json
from human_scene_pilot import restore_saved_integration
from mobiwam.pi05_motion import docks,collaborative_paths
from mobiwam.scene004 import build_minimal_input,candidate_feature_vector,geometry_rule_select
from pi05_candidate_features import feature_record

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();r=a.run
    split=json.loads((r/'data/lineage-split.json').read_text());chosen=[]
    for role,number in [('train',4),('dev',2),('final',2)]:
        pools={task:sorted((x for x in split['parent_groups'] if x['role']==role and x['task']==task),key=lambda x:x['parent_group'])[:number] for task in ('CloseDrawer','CloseSingleDoor')}
        for i in range(number):
            for task in pools:chosen.append(pools[task][i])
    out=r/'inputs/geometry-prefix-v1';out.mkdir(parents=True,exist_ok=False)
    roster=dict(created_at=datetime.now(timezone.utc).isoformat(),rule='parent split fixed before fit; per-task/role sorted identity prefix, task-interleaved; no pi05 outcome selection',groups=chosen,
        counts=dict(train=8,dev=4,final=4,parent_groups=16,families=2),final_outcomes_used=False,known_development=True)
    write_json(out/'roster.json',roster);results=[]
    cache=Path('/share/personal/chensiyu/haokaijiang/MobiWAM/artifacts/MMWAM-OBC-002-DR/DR-v0.4/20260930T115330Z-live-preflight/source-context')
    for index,g in enumerate(chosen):
        groupout=out/g['config_id'];groupout.mkdir();seed=int(re.search(r'seed(\d+)$',g['config_id']).group(1));src=Path(g['source'])
        context_dir=cache/g['task']/f'seed-{seed}';meta=json.loads((context_dir/'manifest.json').read_text())
        context_ok=sha(src/'model.xml')==meta['source_model_sha256'] and sha(src/'integration.npy')==meta['source_integration_sha256']
        if not context_ok:
            write_json(groupout/'input-unavailable.json',dict(reason='existing context source identity mismatch; requires fresh encoder, no feature substitution',source=str(src),role=g['role']))
            results.append(dict(config_id=g['config_id'],status='context_identity_mismatch'));continue
        context=np.load(context_dir/'context.npy',allow_pickle=False);assert context.shape==(4,1024) and sha(context_dir/'context.npy')==meta['context_sha256']
        ref=Reference(argparse.Namespace(output=str(groupout/'native'),task=g['task'],layout=1,style=0,seed=seed,self_test=True,source=str(src),replay_attempt=None,resume_attempt=None,width=640,height=360))
        try:
            restore_saved_integration(ref)
            def prohibited(*args,**kwargs):raise AssertionError('pre-outcome inputs must not step')
            ref.env.step=prohibited
            d=docks(ref);assist=collaborative_paths(ref);write_json(groupout/'D-plan.json',d);write_json(groupout/'A-plan.json',assist)
            rows=[]
            for route,plan in [('E',None),('D',d),('A',assist)]:
                feature=feature_record(ref,route,plan,index)
                if feature is None:continue
                X=np.r_[build_minimal_input(context,candidate_feature_vector(feature['derived'])),np.array([route==x for x in 'EDA'],np.float32)]
                assert X.shape==(1048,) and np.isfinite(X).all();np.save(groupout/(route+'-X.npy'),X.astype(np.float32));rows.append(feature)
            write_json(groupout/'features.json',dict(parent_group=g['parent_group'],config_id=g['config_id'],family_id=g['family_id'],role=g['role'],routes=rows,
                geometry_selection=geometry_rule_select(rows),context_manifest=str(context_dir/'manifest.json'),context_sha256=meta['context_sha256'],
                context_source_identity_exact=True,env_step_calls=0,route_outcomes_accessed=False,policy_queried=False,simulator_oracle=True))
            results.append(dict(config_id=g['config_id'],status='measured',routes=len(rows),role=g['role']));print(json.dumps(results[-1]),flush=True)
        finally:ref.env.close()
    write_json(out/'completed.json',dict(at=datetime.now(timezone.utc).isoformat(),groups=results,new_outcomes=0,final_policy_queries=0,
        feature_scope='initial geometry and geometric motion prefixes, not complete future manipulation safety or learned perception'))
if __name__=='__main__':main()
