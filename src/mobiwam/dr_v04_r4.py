"""R4 immutable six-Source selection and final-checkpoint inference only."""
import json
from pathlib import Path
import numpy as np
import torch
from mobiwam.dr_v04_r3_learning import scale, transform

ROUTES=("E","D","A")
HEADS=("success","collision","progress","base_path_m","terminal_duration_s")
TOLERANCES=(.05,.05,.05,.02,1.)
SCOPE="DR-v0.4_R4_development_selected_route"

def explain_selection(pred, active, hard_valid=(True,True,True)):
    remaining=[i for i,v in enumerate(hard_valid) if v];stages=[]
    if not active or not remaining:return dict(route=None,stages=stages)
    for j,tol in enumerate(TOLERANCES):
        if j not in active:
            stages.append(dict(head=HEADS[j],skipped=True,remaining=[ROUTES[i] for i in remaining]));continue
        values=np.asarray(pred)[remaining,j]
        if not np.isfinite(values).all():raise ValueError("nonfinite supported prediction")
        best=float(max(values) if j in (0,2) else min(values))
        before=list(remaining)
        remaining=[i for i in remaining if (pred[i,j]>=best-tol if j in (0,2) else pred[i,j]<=best+tol)]
        stages.append(dict(head=HEADS[j],tolerance=tol,best=best,before=[ROUTES[i] for i in before],remaining=[ROUTES[i] for i in remaining]))
    return dict(route=ROUTES[min(remaining)],stages=stages,tie_order=list(ROUTES))

def predict(freeze, slot, device):
    r3=Path(freeze["r3_root"]);cp=torch.load(freeze["checkpoint"],map_location="cpu")
    assert cp["step"]==2000
    cfg=cp["config"];z=np.load(freeze["dataset"],allow_pickle=False)
    idx=slot["row_indices"]
    assert list(z["group_id"][idx])==[slot["group_id"]]*3 and list(z["route"][idx])==list(ROUTES)
    X=z["X"][idx].copy()
    xs=scale(X,np.array(cfg["mean"]),np.array(cfg["std"]))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    with torch.no_grad():
        raw=torch.nn.functional.linear(torch.from_numpy(xs).to(device),cp["linear"]["weight"].to(device),cp["linear"]["bias"].to(device)).cpu().numpy()
    p=transform(raw,cfg["head_support"])
    prior=np.load(r3/"predictions/final-offline.npz",allow_pickle=False)
    delta=np.max(abs(p-prior["learned"][idx]))
    if delta>1e-3:raise ValueError(f"final prediction mismatch {delta}")
    planning=json.loads(Path(slot["scientific_row"]["candidate_features"]).read_text())
    valid={x["route"]:bool(x["hard_valid"] and x["features"]["stage_precontact"]==1.) for x in planning["records"]}
    choice=explain_selection(p,cfg["active_heads"],[valid[k] for k in ROUTES])
    original=explain_selection(prior["learned"][idx],cfg["active_heads"],[valid[k] for k in ROUTES])
    if choice["route"]!=original["route"]:raise ValueError("R3/R4 selection differs")
    return dict(group_id=slot["group_id"],head_order=list(HEADS),routes=list(ROUTES),raw_logits=raw.tolist(),predictions=p.tolist(),
                selection=choice,prediction_max_abs_difference_vs_R3=float(delta),R3_selected_route=original["route"],
                checkpoint_sha256=freeze["checkpoint_sha256"],device=str(device),encoder_calls=0,
                source_bound_cached_preoutcome_input=True,scaler_applied_once=True),X,xs
