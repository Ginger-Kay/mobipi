"""One final offline evaluation, same masks and explicit Source denominators."""
import json
from pathlib import Path
import numpy as np
import torch
from mobiwam.dr_v04_r3_learning import *
from dr_v04_r3_prepare import load, write, csvwrite, sha, HEADS

def metrics(pred, raw, y, mask, groups, heads, selection):
    result=[]
    for j,head in enumerate(HEADS):
        idx=selection & mask[:,j] & np.isfinite(pred[:,j])
        if not idx.any():
            result.append(dict(head=head,status="unavailable",rows=0,sources=0));continue
        w=source_weights(idx,groups)[idx];w/=w.sum()
        target=y[idx,j];prob=pred[idx,j];record=dict(head=head,status=heads[j]["status"],rows=int(idx.sum()),sources=len(set(groups[idx])))
        if j<2:
            clipped=np.clip(prob,1e-6,1-1e-6)
            bce=(-(target*np.log(clipped)+(1-target)*np.log1p(-clipped)))
            if raw is not None and heads[j]["status"]=="learnable":
                logits=raw[idx,j].astype(np.float64);bce=np.logaddexp(0.,logits)-target*logits
            record.update(brier=float(w@((prob-target)**2)),BCE=float(w@bce),accuracy=float(w@((prob>=.5)==target)),
                          positive=int((target==1).sum()),negative=int((target==0).sum()),masked_head_loss=float(w@bce))
        else:
            difference=abs(prob-target)/SCALES[j]
            sl=np.where(difference<.1,.5*difference**2/.1,difference-.05)
            record.update(MAE=float(w@abs(prob-target)),units=("fraction" if j==2 else ("m" if j==3 else "s")),
                          masked_head_loss=float(w@sl))
        result.append(record)
    return result

def evaluate_final(run):
    assert not (run/"comparison/metrics.json").exists(),"final evaluation already exists; no repeat"
    cfg=load(run/"training/config.json");result=load(run/"training/result.json")
    assert result["optimizer_steps"]==2000 and sha(result["final_checkpoint"])==result["final_checkpoint_sha256"]
    z=np.load(run/"dataset/supervision.npz",allow_pickle=False)
    X=z["X"];y=z["y"];mask=z["mask"];groups=z["group_id"];tasks=z["task"];routes=z["route"];splits=z["split"]
    heads=cfg["head_support"];active=cfg["active_heads"]
    train=splits=="train";xs=scale(X,np.array(cfg["mean"]),np.array(cfg["std"]))
    cp=torch.load(run/"training/step2000.pt",map_location="cpu")
    # A single full train/validation offline forward of the frozen final weights.
    with torch.no_grad():raw=F.linear(torch.from_numpy(xs),cp["linear"]["weight"].cpu(),cp["linear"]["bias"].cpu()).numpy()
    pred=transform(raw,heads)
    preds,fits=baselines(xs[train],y[train],mask[train],groups[train],tasks[train],routes[train],xs,tasks,routes,heads)
    write(run/"comparison/B3-train-fit.json",dict(fits=fits,objective="sum_Source mean_valid_routes squared error + 1.0*coef_norm2; unpenalized intercept",train_only=True))
    preds["learned"]=pred
    np.savez_compressed(run/"predictions/final-offline.npz",**preds,learned_logits=raw,group_id=groups,route=routes,split=splits,
                        task=tasks,mask=mask,y=y)
    table=[]
    for i in range(len(X)):
        for method,values in preds.items():
            for j,head in enumerate(HEADS):
                table.append(dict(group_id=str(groups[i]),split=str(splits[i]),task=str(tasks[i]),route=str(routes[i]),method=method,head=head,
                                  prediction=float(values[i,j]) if np.isfinite(values[i,j]) else "",
                                  target=float(y[i,j]) if mask[i,j] else "",valid=bool(mask[i,j]),support=heads[j]["status"]))
    csvwrite(run/"predictions/final-offline.csv",table)
    metric_records=[]
    for split in ("train","validation"):
        for task in ("ALL","CloseDrawer","CloseSingleDoor"):
            idx=(splits==split)& ((tasks==task) if task!="ALL" else True)
            for method,values in preds.items():
                if method=="B3_unclipped":continue
                for row in metrics(values,raw if method=="learned" else None,y,mask,groups,heads,idx):
                    metric_records.append(dict(split=split,task=task,method=method,**row))
    # JSON keeps per-metric support; CSV uses a uniform schema.
    write(run/"comparison/metrics.json",metric_records)
    all_keys=list(dict.fromkeys(k for row in metric_records for k in row))
    csvwrite(run/"comparison/metrics.csv",[{k:row.get(k,"") for k in all_keys} for row in metric_records])
    losses=[]
    for split in ("train","validation"):
        for method in ("learned","B1","B2","B3"):
            relevant=[r for r in metric_records if r["split"]==split and r["task"]=="ALL" and r["method"]==method and r["head"] in [HEADS[j] for j in active] and "masked_head_loss" in r]
            losses.append(dict(split=split,method=method,learnable_heads=len(relevant),
                               masked_loss=float(np.mean([r["masked_head_loss"] for r in relevant])) if relevant else None))
    write(run/"comparison/masked-loss.json",losses)
    records=load(run/"dataset/route-records.json")
    group_index={g:np.flatnonzero(groups==g) for g in sorted(set(groups))}
    complete_train=[g for g,idx in group_index.items() if splits[idx[0]]=="train" and mask[idx].all()]
    assert all(list(routes[idx])==["E","D","A"] for idx in group_index.values())
    if complete_train:
        aggregates=[]
        for k,route in enumerate(("E","D","A")):
            v=np.stack([y[group_index[g][k]] for g in complete_train])
            aggregates.append((-float(v[:,0].mean()),float(v[:,1].mean()),-float(v[:,2].mean()),float(v[:,3].mean()),float(v[:,4].mean()),k))
        best=min(range(3),key=lambda k:aggregates[k]);best_fixed=("E","D","A")[best]
    else:best=None;best_fixed=None
    write(run/"comparison/train-best-fixed.json",dict(route=best_fixed,status="available" if best is not None else "unavailable",
           sources=complete_train,used_sources=len(complete_train),original_train_denominator=24,
           criterion="complete E/D/A common-known five targets; success up/collision down/progress up/path down/terminal duration down/E<D<A"))
    selections=[];selection_map={}
    for g,idx in group_index.items():
        chosen={"fixed_E":0,"fixed_D":1,"fixed_A":2,"train_best_fixed":best,
                "geometry":("E","D","A").index(records[idx[0]]["geometry_preoutcome_choice"])}
        for method in ("learned","B1","B2","B3"):chosen[method]=select(preds[method][idx],active)
        chosen["oracle"]=min(range(3),key=lambda k:(-y[idx[k],0],y[idx[k],1],-y[idx[k],2],y[idx[k],3],y[idx[k],4],k)) if mask[idx].all() else None
        selection_map[g]=chosen
        for method,k in chosen.items():
            i=idx[k] if k is not None else None
            selections.append(dict(group_id=g,split=str(splits[idx[0]]),task=str(tasks[idx[0]]),method=method,
                route=str(routes[i]) if i is not None else "",status="available" if i is not None else "unavailable",
                success=float(y[i,0]) if i is not None and mask[i,0] else "",
                collision=float(y[i,1]) if i is not None and mask[i,1] else "",
                progress=float(y[i,2]) if i is not None and mask[i,2] else "",
                base_path_m=float(y[i,3]) if i is not None and mask[i,3] else "",
                terminal_duration_s=float(y[i,4]) if i is not None and mask[i,4] else ""))
    csvwrite(run/"comparison/route-selections.csv",selections)
    summary=[];paired=[]
    methods=["fixed_E","fixed_D","fixed_A","train_best_fixed","geometry","B1","B2","B3","learned","oracle"]
    for split in ("train","validation"):
        gs=[g for g,idx in group_index.items() if splits[idx[0]]==split];denom=len(gs)
        for method in methods:
            row=dict(split=split,method=method,source_denominator=denom,selection_sources=sum(selection_map[g][method] is not None for g in gs))
            for j,head in enumerate(HEADS):
                known=[g for g in gs if selection_map[g][method] is not None and mask[group_index[g][selection_map[g][method]],j]]
                vals=[float(y[group_index[g][selection_map[g][method]],j]) for g in known]
                row[head]=dict(known_sources=len(known),coverage=len(known)/denom,missing_sources=[g for g in gs if g not in known],
                               sum=float(sum(vals)) if j<2 and vals else None,mean=float(np.mean(vals)) if vals else None)
            summary.append(row)
        for comparator in methods:
            if comparator=="learned":continue
            for j,head in enumerate(HEADS):
                common=[];differences=[]
                for g in gs:
                    k=selection_map[g]["learned"];l=selection_map[g][comparator]
                    if k is None or l is None:continue
                    i=group_index[g][k];q=group_index[g][l]
                    if mask[i,j] and mask[q,j]:
                        common.append(g);differences.append(float(y[i,j]-y[q,j]))
                paired.append(dict(split=split,method="learned",comparator=comparator,head=head,
                    source_denominator=denom,common_sources=len(common),common_group_ids=common,
                    sum_difference=float(sum(differences)) if differences else None,
                    mean_difference=float(np.mean(differences)) if differences else None,
                    missing_sources=[g for g in gs if g not in common]))
    write(run/"comparison/selector-summary.json",summary);write(run/"comparison/paired-differences.json",paired)
    write(run/"predictions/evaluation-receipt.json",dict(checkpoint_sha256=result["final_checkpoint_sha256"],offline_forward_calls=1,
          train_rows=72,validation_rows=36,validation_sources=12,selector_active_heads=[HEADS[j] for j in active],
          skipped_heads=[HEADS[j] for j in range(5) if j not in active],sealed_test_read=False,
          feature_encoder_invocations=0,baseline_fit_train_only=True,formal_train_ready=False))
    print(json.dumps(dict(evaluation="completed",masked_loss=losses,train_best_fixed=best_fixed)),flush=True)
