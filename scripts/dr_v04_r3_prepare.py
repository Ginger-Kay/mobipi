"""Reclassify saved evidence only; no env.step, encoder, sealed-test access."""
import argparse, csv, json, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import numpy as np
import mujoco
from mobiwam.contact_rules import FINGER_PAD_PAIR, RULE_VERSION, allowed_contact
from mobiwam.reference_collision import SweptGeometry
from mobiwam.task_video_identity import source_model

HEADS = ["success", "collision", "progress", "base_path_m", "terminal_duration_s"]
OLD = ["success", "failure", "progress", "base_path_m", "completion_time_s"]

def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, x): Path(p).write_text(json.dumps(x, indent=2, allow_nan=False)+"\n")
def csvwrite(p, rows):
    with Path(p).open("w") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def recheck(arg):
    row, out = arg
    p=Path(row["attempt"]); old=load(row["audit"])
    modelpath=p.parent.parent/"model.xml"
    model=source_model(str(modelpath), sha(modelpath))
    native=load(p/"task-video-manifest.json")["binding"]["native"]
    target=native["fixture_name"]
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(model.ngeom)]
    ids=[names.index(n) for n in FINGER_PAD_PAIR]
    physics={k:getattr(model,k).copy() for k in ("geom_contype","geom_conaffinity","geom_friction","geom_solref","geom_solimp")}
    with np.load(p/"formal-native-substeps.npz",allow_pickle=False) as z:
        states=z["qpos"]; phases=z["phases"].tolist()
    geom=SweptGeometry(model,target_prefix=target,margin=.0005)
    assert not any(set(pair)==set(ids) for pair in geom.pairs)
    sweep=geom.path(states,phases)
    for k,v in physics.items():np.testing.assert_array_equal(v,getattr(model,k))
    oldstop=load(p/"formal-substep-stop.json")
    contacts=oldstop["contacts"]
    still_forbidden=[c for c in contacts if not allowed_contact(c.get("geom1"),c.get("geom2"),oldstop["phase"],target)]
    result=dict(group_id=row["group_id"],route=row["route"],attempt=str(p),
                contact_rule_version=RULE_VERSION,old_audit=row["audit"],
                old_stop=oldstop,remaining_stop_contacts=still_forbidden,
                full_recorded_native_path_sweep=sweep,
                pair_binding=dict(names=sorted(FINGER_PAD_PAIR),geom_ids=ids),
                physical_parameters_unchanged=True,record_integrity="reused_closed_R2_receipt",
                old_rule_truncated=True,full_new_rule_outcome_known=False,
                saved_prefix_safety_status="pass" if sweep["valid"] and not still_forbidden else "not_passed",
                trajectory_scope="entire saved prefix including partial native terminal substep; no continuation",
                replay_reused=old["replay_reproducible"])
    write(Path(out)/f'{row["group_id"]}-{row["route"]}.json',result)
    print(json.dumps(dict(group=row["group_id"],route=row["route"],sweep=sweep)),flush=True)
    return result

def main():
    a=argparse.ArgumentParser();a.add_argument("--run",type=Path,required=True);a.add_argument("--r2",type=Path,required=True)
    args=a.parse_args();run=args.run;r2=args.r2
    snap=r2/"gate/snapshot-20261002T221056315438Z"
    rows=load(snap/"route-outcomes.json")
    assert len(rows)==108 and len({r["group_id"] for r in rows})==36
    assert {r["split"] for r in rows}=={"train","validation"}
    with np.load(snap/"paired-supervision-candidate.npz",allow_pickle=False) as z:
        X=z["X"].copy(); old_y=z["y"].copy(); group=z["group_id"];route=z["route"];split=z["split"]
    assert X.shape==(108,1045) and np.isfinite(X).all()
    assert list(zip(group,route))==[(r["group_id"],r["route"]) for r in rows]
    meta=load(snap/"paired-supervision-candidate.json")
    assert sha(snap/"paired-supervision-candidate.npz")==meta["sha256"]
    with (r2/"gate/safety-review-20261003T024614Z/human-review-template.csv").open() as f: approved=list(csv.DictReader(f))
    assert len(approved)==9
    by={(r["group_id"],r["route"]):r for r in rows}
    for review in approved:
        row=by[review["group_id"],review["route"]]
        assert row["video"]==review["video"] and row["video_sha256"]==review["video_sha256"]
        native=load(Path(row["attempt"])/"task-video-manifest.json")
        assert native["files"]["original.mp4"]["sha256"]==review["video_sha256"]
        assert native["binding"]["group_id"]==review["group_id"] and native["binding"]["route"]==review["route"]
        assert Path(review["video"]).stat().st_size==native["files"]["original.mp4"]["size"]
        review.update(human_review="approved",full_video_watched="true",reviewed_by="researcher",reviewed_at="2026-10-03T04:52:09+00:00")
        row["human_review"]="approved"
    csvwrite(run/"audit/human-review-approved.csv",approved)
    affected=[r for r in rows if r["raw_reason"]=="native_forbidden_contact_stop"]
    assert {(r["group_id"],r["route"]) for r in affected}=={(r["group_id"],r["route"]) for r in approved}
    jobs=[]
    for row in affected:
        p=run/"audit"/f'{row["group_id"]}-{row["route"]}.json'
        if p.exists():assert load(p)["contact_rule_version"]==RULE_VERSION
        else:jobs.append((row,str(run/"audit")))
    with ProcessPoolExecutor(max_workers=3) as pool:list(pool.map(recheck,jobs))
    y=old_y[:,[0,2,1,3,4]].copy(); mask=np.isfinite(y); reasons=np.full((108,5),"known_complete_terminal_record",dtype="<U80")
    diff=[]
    for i,row in enumerate(rows):
        audit=load(row["audit"])
        assert audit["source_input_sha256"]==row["source_input_sha256"] and audit["replay_reproducible"]
        assert audit["group_id"]==row["group_id"] and audit["route"]==row["route"]
        affected_here=row["raw_reason"]=="native_forbidden_contact_stop"
        if affected_here:
            y[i,:]=np.nan; mask[i,:]=False;reasons[i,:]="old_exempt_rule_truncation_new_terminal_outcome_unknown"
            safety=load(run/"audit"/f'{row["group_id"]}-{row["route"]}.json')["saved_prefix_safety_status"]
        else:
            assert audit["actual_swept_geometry"]["valid"] and not audit["realized_contact_stop"]
            safety="pass_reused_stricter_R2_receipt"
        row.update(record_integrity="pass_reused_receipt",safety_status=safety,
                   termination_reason=row["raw_reason"],supervision_eligibility="partial_or_none" if affected_here else "all_heads",
                   label_valid_mask=mask[i].tolist(),rule_version=RULE_VERSION)
        for j,head in enumerate(HEADS):
            diff.append(dict(group_id=row["group_id"],split=row["split"],route=row["route"],head=head,
                             old=float(old_y[i,[0,2,1,3,4][j]]),new=float(y[i,j]) if mask[i,j] else "",
                             valid=bool(mask[i,j]),reason=str(reasons[i,j])))
    availability=[]
    for s in ("train","validation"):
        for task in ("ALL","CloseDrawer","CloseSingleDoor"):
            for r in ("ALL","E","D","A"):
                selected=np.array([x["split"]==s and (task=="ALL" or x["task"]==task) and (r=="ALL" or x["route"]==r) for x in rows])
                for j,head in enumerate(HEADS):
                    idx=selected & mask[:,j]
                    availability.append(dict(split=s,task=task,route=r,head=head,valid_rows=int(idx.sum()),
                        independent_sources=len(set(group[idx])),positive=int((y[idx,j]==1).sum()) if j<2 else "",
                        negative=int((y[idx,j]==0).sum()) if j<2 else "",missing_rows=int((selected & ~mask[:,j]).sum()),
                        missing_reason="old_exempt_rule_truncation" if (selected & ~mask[:,j]).any() else ""))
    np.savez_compressed(run/"dataset/supervision.npz",X=X,y=y,mask=mask,reason=reasons,group_id=group,route=route,split=split,
                        task=np.array([r["task"] for r in rows]),old_y=old_y)
    csvwrite(run/"dataset/label-availability.csv",availability);csvwrite(run/"dataset/label-diff.csv",diff)
    write(run/"dataset/route-records.json",rows)
    schema=dict(created_at=datetime.now(timezone.utc).isoformat(),heads=HEADS,X_shape=[108,1045],y_shape=[108,5],
                label_definitions=dict(success="native checker at complete terminal stop; stall is failure",
                    collision="existing irreversible_failure=collision adapter; witnessed forbidden native contact positive, full strict saved path negative; future of censored prefix unknown",
                    progress="clip(1-native target door opening,0,1) at complete terminal outcome",
                    base_path_m="sum native recorded base XY displacements through complete termination, meters",
                    terminal_duration_s="native simulator time through complete termination, including stall; inherited completion_time_s was elapsed-to-stop, not success-conditional"),
                control_hz=20,horizon_steps=2400,horizon_seconds=120,path_scale_m=2.0,time_scale_s=120.0,
                normalization="no upper label clipping",feature_receipt=str(r2/"gate/preoutcome-inputs-receipt.json"),
                old_labels_order=meta["labels_order"],old_snapshot=str(snap),old_dataset_sha256=meta["sha256"],
                dataset_sha256=sha(run/"dataset/supervision.npz"),formal_train_ready=False,data_ready_for_development=True,
                sealed_test_read=False,valid_labels=int(mask.sum()),valid_rows=int(mask.any(1).sum()),
                nine_approved_videos=9,encoder_invocations=0,physical_change=False)
    write(run/"dataset/schema.json",schema)
    for p in (run/"dataset").iterdir():p.chmod(0o444)
    print(json.dumps(schema),flush=True)

if __name__=="__main__":main()
