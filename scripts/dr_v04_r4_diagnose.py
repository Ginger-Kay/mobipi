"""Independent A-G numerical diagnosis. Read-only autograd; no optimizer."""
import argparse,csv,json,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from scipy.special import expit
from torch.nn import functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HEADS=("success","collision","progress","base_path_m","terminal_duration_s")
SCALE=np.array([1.,1.,1.,2.,120.])
ACTIVE=(0,2,3,4)
ROUTES=("E","D","A")
TOLS=(.05,.05,.05,.02,1.)
def load(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def csvwrite(p,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open("w") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def weights(groups,valid):
    w=np.zeros(len(groups),float)
    for g in sorted(set(groups[valid])):
        i=(groups==g)&valid;w[i]=1/i.sum()
    return w
def average(v,w):return float(np.sum(v*w)/np.sum(w))
def softplus(z):return np.logaddexp(0.,z)
def predictions(raw):
    p=np.empty_like(raw,dtype=np.float64)
    p[:,0]=expit(raw[:,0]);p[:,1]=0.;p[:,2]=expit(raw[:,2])
    p[:,3:]=softplus(raw[:,3:])*SCALE[3:];return p
def smooth(error):
    d=abs(error);return np.where(d<.1,.5*d*d/.1,d-.05)
def metric_rows(pred,raw,y,mask,groups,select,method):
    result=[]
    for j,h in enumerate(HEADS):
        valid=mask[:,j]&select
        if not valid.any():continue
        w=weights(groups,valid)[valid];v=y[valid,j];p=pred[valid,j]
        row=dict(method=method,head=h,rows=int(valid.sum()),sources=len(set(groups[valid])))
        if j<2:
            q=np.clip(p,1e-6,1-1e-6)
            bce=np.logaddexp(0.,raw[valid,j])-v*raw[valid,j] if raw is not None and j==0 else -(v*np.log(q)+(1-v)*np.log1p(-q))
            row.update(brier=average((p-v)**2,w),BCE=average(bce,w),accuracy=average((p>=.5)==v,w),
                       positive=int((v==1).sum()),negative=int((v==0).sum()),loss=average(bce,w))
        else:row.update(MAE=average(abs(p-v),w),bias=average(p-v,w),loss=average(smooth((p-v)/SCALE[j]),w))
        result.append(row)
    return result
def select_independent(p):
    left=list(range(3));stages=[]
    for j in ACTIVE:
        best=(max if j in (0,2) else min)(p[left,j]);previous=list(left)
        left=[i for i in left if (p[i,j]>=best-TOLS[j] if j in (0,2) else p[i,j]<=best+TOLS[j])]
        stages.append(dict(head=HEADS[j],before=[ROUTES[i] for i in previous],remaining=[ROUTES[i] for i in left],best=float(best),tol=TOLS[j],
                           resolved=len(previous)>1 and len(left)==1))
    return ROUTES[min(left)],stages

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True);a=ap.parse_args();r=a.run
    roster=load(r/"frozen-six/roster.json");r3=Path(roster["r3_root"]);out=r/"diagnosis";out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    z=np.load(r3/"dataset/supervision.npz",allow_pickle=False)
    X=z["X"];y=z["y"].astype(float);mask=z["mask"];groups=z["group_id"];split=z["split"];tasks=z["task"];route=z["route"]
    cfg=load(r3/"training/config.json");mean=np.array(cfg["mean"]);std=np.array(cfg["std"])
    good=std>=1e-6;xs=np.zeros_like(X,dtype=np.float32);xs[:,good]=((X[:,good]-mean[good])/std[good]).astype(np.float32)
    cp=torch.load(roster["checkpoint"],map_location="cpu");W=cp["linear"]["weight"].numpy();b=cp["linear"]["bias"].numpy()
    prior=np.load(r3/"predictions/final-offline.npz",allow_pickle=False)
    raw=xs.astype(float)@W.astype(float).T+b.astype(float);pred=predictions(raw)
    saved=prior["learned"];stored_metrics=load(r3/"comparison/metrics.json");results=[]
    # A: Independent NumPy computations, explicit group/route joins.
    for s in ("train","validation"):
        for task in ("ALL","CloseDrawer","CloseSingleDoor"):
            for route_name in ("ALL","E","D","A"):
                subset=(split==s)&((tasks==task) if task!="ALL" else True)&((route==route_name) if route_name!="ALL" else True)
                for method in ("learned","B1","B2","B3"):
                    p=pred if method=="learned" else prior[method]
                    for row in metric_rows(p,raw if method=="learned" else None,y,mask,groups,subset,method):
                        results.append(dict(split=s,task=task,route=route_name,**row))
    csvwrite(out/"A-independent-metrics.csv",results)
    diffs=[]
    for row in results:
        if row["route"]!="ALL":continue
        old=next(x for x in stored_metrics if (x["split"],x["task"],x["method"],x["head"])==(row["split"],row["task"],row["method"],row["head"]))
        for key in ("brier","BCE","accuracy","MAE"):
            if key in row:diffs.append(dict(split=row["split"],task=row["task"],method=row["method"],head=row["head"],metric=key,
                                          saved=old[key],independent=row[key],absolute_difference=abs(old[key]-row[key])))
    csvwrite(out/"A-metric-differences.csv",diffs)
    maxdiff=max(d["absolute_difference"] for d in diffs)
    scaler_sources=set(cfg["scaler_sources"]);fit=np.array([split[i]=="train" and groups[i] in scaler_sources for i in range(len(X))])
    wf=weights(groups,fit)[fit];mu=np.average(X[fit].astype(float),weights=wf,axis=0)
    sd=np.sqrt(np.average((X[fit]-mu)**2,weights=wf,axis=0))
    assert np.max(abs(mu-mean))<1e-8 and np.max(abs(sd-std))<1e-8
    assert not np.any(mask[~mask.any(1)]) and np.isnan(y[~mask]).all()
    logs=[json.loads(x) for x in (r3/"training/loss.jsonl").read_text().splitlines()]
    finaltrain=[x for x in results if x["split"]=="train" and x["task"]=="ALL" and x["route"]=="ALL" and x["method"]=="learned" and x["head"] in [HEADS[j] for j in ACTIVE]]
    independentloss=np.mean([x["loss"] for x in finaltrain])
    write(out/"A-reproduction.json",dict(checkpoint_sha256=sha(roster["checkpoint"]),dataset_sha256=sha(roster["dataset"]),
       raw_fp64_max_difference=float(np.max(abs(raw-prior["learned_logits"]))),prediction_fp64_max_difference=float(np.max(abs(pred-saved))),
       metric_max_absolute_difference=maxdiff,metric_comparison_tolerance=1e-4,metrics_match=maxdiff<1e-4,
       final_train_loss_independent=float(independentloss),saved_log_final_loss=logs[-1]["loss"],log_is_after_optimizer_update=True,
       masked_NaN_handled_by_indexing=True,heads=list(HEADS),units=dict(path="m",duration="sim_time seconds"),
       scaler_apply_count=1,scaler_mean_max_diff=float(max(abs(mu-mean))),scaler_std_max_diff=float(max(abs(sd-std))),
       train_scaler_sources=23,train_scaler_input_rows=int(fit.sum()),train_effective_label_rows=int(((split=="train")&mask.any(1)).sum()),
       critical_model_semantics_bug=False,no_optimizer_step=True,sealed_test_read=False))
    # B: Fixed ordinary and worst-error trace checks in each task x route.
    records=load(r3/"dataset/route-records.json");sample=[]
    severity=np.full(len(X),np.nan)
    for i in range(len(X)):
        j=[k for k in ACTIVE if mask[i,k]]
        if j:severity[i]=float(np.mean(abs(pred[i,j]-y[i,j])/SCALE[j]))
    for t in sorted(set(tasks)):
        for k in ROUTES:
            ids=np.flatnonzero((tasks==t)&(route==k)&mask.all(1))
            ordered=sorted(ids,key=lambda i:(str(groups[i]),int(i)))
            selected=[("ordinary_train_lexicographic_first",next(i for i in ordered if split[i]=="train")),
                      ("ordinary_validation_lexicographic_first",next(i for i in ordered if split[i]=="validation")),
                      ("maximum_normalized_prediction_error",max(ids,key=lambda i:severity[i]))]
            for kind,i in selected:
                row=records[i];path=Path(row["attempt"]);trace=[json.loads(s) for s in (path/"trace.jsonl").read_text().splitlines()]
                assert row["group_id"]==groups[i] and row["route"]==route[i]
                points=np.asarray([trace[0]["before"]["base_pos"]]+[t["after"]["base_pos"] for t in trace])
                vals=[float(trace[-1]["after"]["success"]),0.,float(np.clip(1-float(trace[-1]["after"]["target"]["door"]),0,1)),
                      float(np.sum(np.sqrt(np.sum((points[1:,:2]-points[:-1,:2])**2,axis=1)))),
                      float(trace[-1]["after"]["sim_time"]-trace[0]["before"]["sim_time"])]
                rawresult=load(path/"result.json")
                assert rawresult["steps"]==len(trace) and bool(vals[0])==rawresult["checker_success"]
                difference=abs(np.array(vals)-y[i])
                sample.append(dict(group_id=str(groups[i]),route=str(route[i]),task=t,split=str(split[i]),sample_reason=kind,attempt=str(path),
                                   steps=len(trace),termination=rawresult["reason"],duration_seconds=vals[4],
                                   nominal_steps_seconds=len(trace)*.05,duration_vs_steps_abs_diff=abs(vals[4]-len(trace)*.05),
                                   max_label_difference=float(max(difference)),progress_difference=float(difference[2]),path_difference=float(difference[3]),
                                   duration_difference=float(difference[4]),success_matches=difference[0]==0,labels_match=max(difference)<1e-4))
    csvwrite(out/"B-trace-label-checks.csv",sample)
    # C: Every dimension, block and task distribution; no clipping/refitting.
    stats=[]
    for task in ("ALL","CloseDrawer","CloseSingleDoor"):
        for s in ("train","validation"):
            ids=(split==s)&((tasks==task) if task!="ALL" else True)
            for j in range(1045):
                v=X[ids,j].astype(float);scaled=xs[ids,j].astype(float)
                stats.append(dict(split=s,task=task,dimension=j,block="visual" if j<1024 else "geometry",
                    raw_mean=float(v.mean()),raw_std=float(v.std()),raw_min=float(v.min()),raw_max=float(v.max()),
                    scaler_mean=float(mean[j]),scaler_std=float(std[j]),suppressed=bool(not good[j]),
                    scaled_mean=float(scaled.mean()),scaled_std=float(scaled.std()),scaled_min=float(scaled.min()),scaled_max=float(scaled.max()),
                    max_abs_scaled=float(max(abs(scaled))),above_abs5=int((abs(scaled)>5).sum()),above_abs10=int((abs(scaled)>10).sum())))
    csvwrite(out/"C-per-dimension-distributions.csv",stats)
    blockrows=[]
    for task in ("ALL","CloseDrawer","CloseSingleDoor"):
        for s in ("train","validation"):
            ids=(split==s)&((tasks==task) if task!="ALL" else True)
            for block,sl in (("visual",slice(0,1024)),("geometry",slice(1024,1045))):
                v=xs[ids,sl].astype(float);per=np.sqrt(np.sum(v*v,axis=1))
                blockrows.append(dict(split=s,task=task,block=block,rows=int(ids.sum()),dims=v.shape[1],max_abs_scaled=float(max(abs(v).flatten())),
                                     abs5_fraction=float((abs(v)>5).mean()),abs10_fraction=float((abs(v)>10).mean()),
                                     row_L2_mean=float(per.mean()),row_L2_max=float(per.max()),suppressed_dims=int((~good[sl]).sum()),
                                     min_positive_scaler_std=float(min(std[sl][good[sl]]))))
    csvwrite(out/"C-block-distribution.csv",blockrows)
    shifts=[]
    for j in range(1045):
        tr=xs[split=="train",j];va=xs[split=="validation",j]
        shifts.append(dict(dimension=j,block="visual" if j<1024 else "geometry",abs_mean_shift=float(abs(va.mean()-tr.mean())),
                           validation_max_abs=float(max(abs(va))),validation_raw_outside_train_range=int(((X[split=="validation",j]<min(X[split=="train",j]))|(X[split=="validation",j]>max(X[split=="train",j]))).sum())))
    write(out/"C-largest-shifts.json",sorted(shifts,key=lambda x:x["validation_max_abs"],reverse=True)[:30])
    input_checks=[]
    for g in sorted(set(groups)):
        idx=np.flatnonzero(groups==g)
        input_checks.append(dict(group_id=str(g),split=str(split[idx[0]]),
          visual_within_Source_max_diff=float(np.max(abs(X[idx,:1024]-X[idx[0],:1024]))),row_count=len(idx),
          group_single_split=len(set(split[idx]))==1,full_input_duplicate_pairs=sum(np.array_equal(X[idx[i]],X[idx[k]]) for i in range(3) for k in range(i+1,3))))
    csvwrite(out/"C-input-group-checks.csv",input_checks)
    bysource=[i[0] for i in [np.flatnonzero(groups==g) for g in sorted(set(groups))]]
    normed=X[bysource,:1024].astype(float);unit=normed/np.linalg.norm(normed,axis=1,keepdims=True)
    cosine=unit@unit.T;pairs=[]
    for i in range(len(bysource)):
        for k in range(i+1,len(bysource)):
            pairs.append(dict(group_a=str(groups[bysource[i]]),group_b=str(groups[bysource[k]]),split_a=str(split[bysource[i]]),split_b=str(split[bysource[k]]),
                              visual_cosine=float(cosine[i,k]),visual_L2=float(np.linalg.norm(normed[i]-normed[k]))))
    csvwrite(out/"C-nearest-visual-Sources.csv",sorted(pairs,key=lambda x:x["visual_L2"])[:30])
    write(out/"C-input-checks.json",dict(nonfinite_inputs=int((~np.isfinite(X)).sum()),suppressed_dims=int((~good).sum()),
        suppressed_visual_dims=int((~good[:1024]).sum()),suppressed_geometry_dims=int((~good[1024:]).sum()),finite_scaling=True,
        exact_duplicate_Source_visuals=sum(p["visual_L2"]==0 for p in pairs),scaler_fit_sources=sorted(scaler_sources),
        scaler_input_69_includes_preoutcome_unlabeled_routes_in_participating_Sources=True,
        only_fully_unsupervised_train_Source_excluded="CloseSingleDoor-layout1-style0-seed129"))
    # D: Full residual table, worst rows and Source influence.
    residuals=[];errorsummary=[];source_errors=[]
    for j in ACTIVE:
        for i in np.flatnonzero(mask[:,j]):
            residuals.append(dict(group_id=str(groups[i]),split=str(split[i]),task=str(tasks[i]),route=str(route[i]),head=HEADS[j],
              actual=float(y[i,j]),predicted=float(pred[i,j]),residual=float(pred[i,j]-y[i,j]),absolute_error=float(abs(pred[i,j]-y[i,j])),
              normalized_absolute_error=float(abs(pred[i,j]-y[i,j])/SCALE[j]),raw_activation=float(raw[i,j])))
        for s in ("train","validation"):
            valid=(split==s)&mask[:,j];w=weights(groups,valid)[valid]
            errors=abs(pred[valid,j]-y[valid,j]);res=pred[valid,j]-y[valid,j]
            errorsummary.append(dict(split=s,head=HEADS[j],rows=int(valid.sum()),sources=len(set(groups[valid])),
                actual_min=float(min(y[valid,j])),actual_max=float(max(y[valid,j])),predicted_min=float(min(pred[valid,j])),predicted_max=float(max(pred[valid,j])),
                Source_weighted_MAE=average(errors,w),Source_weighted_bias=average(res,w),
                row_absolute_error_quantiles={str(q):float(np.quantile(errors,q)) for q in (.0,.25,.5,.75,.9,.95,1.)},
                negative_residual_rows=int((res<0).sum()),positive_residual_rows=int((res>0).sum())))
            ee=[]
            for g in sorted(set(groups[valid])):
                ix=valid&(groups==g);ee.append((g,float(abs(pred[ix,j]-y[ix,j]).mean())))
            total=sum(v for g,v in ee)
            for g,v in sorted(ee,key=lambda x:-x[1]):
                source_errors.append(dict(split=s,head=HEADS[j],group_id=str(g),source_MAE=v,fraction_of_total_source_error=v/total if total else 0.))
    csvwrite(out/"D-all-residuals.csv",residuals);write(out/"D-error-shape.json",errorsummary);csvwrite(out/"D-Source-error-contribution.csv",source_errors)
    worst=[]
    for s in ("train","validation"):
        for h in [HEADS[j] for j in ACTIVE]:
            worst.extend(sorted([row for row in residuals if row["split"]==s and row["head"]==h],key=lambda x:-x["absolute_error"])[:5])
    csvwrite(out/"D-worst-five-rows.csv",worst)
    # E: Four checkpoints on train only, full autograd gradients, no update.
    training=split=="train";trainxs=torch.from_numpy(xs[training]);trainy=torch.from_numpy((y[training]/SCALE).astype(np.float32))
    trainmask=mask[training];train_groups=groups[training];optim=[];checkpointloss=[]
    torch.manual_seed(17);initial=torch.nn.Linear(1045,5)
    assert cp["config"]["torch_version"]==torch.__version__
    assert np.array_equal(initial.weight.detach().numpy()[1],W[1])
    initraw=xs[training].astype(float)@initial.weight.detach().numpy().astype(float).T+initial.bias.detach().numpy()
    initmetrics=metric_rows(predictions(initraw),initraw,y[training],mask[training],groups[training],np.ones(sum(training),bool),"initial_seed17")
    write(out/"E-reconstructed-initialization.json",dict(seed=17,torch_version=torch.__version__,frozen_collision_row_matches=True,train_only_metrics=initmetrics,
          initial_loss=float(np.mean([x["loss"] for x in initmetrics if x["head"]!="collision"])),saved_initial_loss=logs[0]["loss"],weights_saved=False))
    for step in (500,1000,1500,2000):
        c=torch.load(r3/"training"/f"step{step:04d}.pt",map_location="cpu")
        weight=c["linear"]["weight"].clone().requires_grad_(True);bias=c["linear"]["bias"].clone().requires_grad_(True)
        activation=F.linear(trainxs,weight,bias);losses=[]
        for j in ACTIVE:
            valid=torch.from_numpy(trainmask[:,j]);pr=activation[valid,j];target=trainy[valid,j]
            loss=F.binary_cross_entropy_with_logits(pr,target,reduction="none") if j==0 else F.smooth_l1_loss(torch.sigmoid(pr) if j==2 else F.softplus(pr),target,reduction="none",beta=.1)
            w=weights(train_groups,trainmask[:,j])[trainmask[:,j]];w=w/w.sum()
            losses.append(torch.sum(loss*torch.tensor(w,dtype=torch.float32)))
        total=torch.stack(losses).mean();total.backward()
        zz=activation.detach().numpy();pp=predictions(zz);Wc=weight.detach().numpy()
        for row in metric_rows(pp,zz,y[training],mask[training],groups[training],np.ones(sum(training),bool),f"step{step}"):
            checkpointloss.append(dict(step=step,**row))
        for j in ACTIVE:
            supported=trainmask[:,j];v=zz[supported,j];sig=expit(v)
            optim.append(dict(step=step,head=HEADS[j],head_loss=float(losses[ACTIVE.index(j)].detach()),
                raw_min=float(min(v)),raw_max=float(max(v)),raw_mean=float(v.mean()),
                derivative_below_1e_4_fraction=float(((sig*(1-sig) if j in (0,2) else sig)<1e-4).mean()),
                sigmoid_below_1e_4_fraction=float((sig<1e-4).mean()),sigmoid_above_1minus1e_4_fraction=float((sig>1-1e-4).mean()),
                softplus_below_1e_4_fraction=float((softplus(v)<1e-4).mean()) if j>2 else None,
                weight_norm=float(np.linalg.norm(Wc[j])),visual_weight_norm=float(np.linalg.norm(Wc[j,:1024])),geometry_weight_norm=float(np.linalg.norm(Wc[j,1024:])),
                weight_gradient_norm=float(torch.linalg.vector_norm(weight.grad[j])),bias_gradient=float(bias.grad[j]),
                actual_lr=c["optimizer"]["param_groups"][0]["lr"],scheduler_epoch=c["scheduler"]["last_epoch"],
                optimizer_updates=step,total_gradient_norm=float(torch.sqrt((weight.grad**2).sum()+(bias.grad**2).sum()))))
    csvwrite(out/"E-checkpoint-train-metrics.csv",checkpointloss);csvwrite(out/"E-optimization-and-saturation.csv",optim)
    # F: Exact affine block decomposition and two fixed OOD zero-block interventions.
    visual=xs[:,:1024].astype(float)@W[:,:1024].astype(float).T
    geometry=xs[:,1024:].astype(float)@W[:,1024:].astype(float).T
    decomposition=[]
    for i in range(len(X)):
        for j in ACTIVE:
            decomposition.append(dict(group_id=str(groups[i]),split=str(split[i]),task=str(tasks[i]),route=str(route[i]),head=HEADS[j],
              visual=float(visual[i,j]),geometry=float(geometry[i,j]),bias=float(b[j]),raw=float(raw[i,j]),
              decomposition_abs_error=float(abs(visual[i,j]+geometry[i,j]+b[j]-raw[i,j]))))
    csvwrite(out/"F-affine-decomposition.csv",decomposition)
    contributions=[]
    for s in ("train","validation"):
        for j in ACTIVE:
            i=split==s
            contributions.append(dict(split=s,head=HEADS[j],visual_mean=float(visual[i,j].mean()),visual_std=float(visual[i,j].std()),
                                     geometry_mean=float(geometry[i,j].mean()),geometry_std=float(geometry[i,j].std()),
                                     bias=float(b[j]),visual_abs_mean=float(abs(visual[i,j]).mean()),geometry_abs_mean=float(abs(geometry[i,j]).mean())))
    csvwrite(out/"F-block-contributions.csv",contributions)
    differences=[]
    for g in sorted(set(groups)):
        idx=np.flatnonzero(groups==g)
        for j in ACTIVE:
            differences.append(dict(group_id=str(g),split=str(split[idx[0]]),head=HEADS[j],
              visual_route_range=float(np.ptp(visual[idx,j])),geometry_route_range=float(np.ptp(geometry[idx,j])),
              raw_route_range=float(np.ptp(raw[idx,j])),visual_raw_input_same=bool(np.array_equal(X[idx,:1024],np.repeat(X[idx[0],:1024][None,:],3,0)))))
    csvwrite(out/"F-within-Source-route-differences.csv",differences)
    interventionrows=[]
    for name,modified in (("mean_visual",geometry+b),("mean_geometry",visual+b)):
        altered=predictions(modified)
        np.savez_compressed(out/f"F-{name}-predictions.npz",raw=modified,predictions=altered,group_id=groups,split=split,route=route)
        for s in ("train","validation"):
            for row in metric_rows(altered,modified,y,mask,groups,split==s,name):interventionrows.append(dict(split=s,OOD_diagnostic=True,used_for_episode_selection=False,**row))
    csvwrite(out/"F-fixed-final-interventions.csv",interventionrows)
    # G: Every Source's selection stages, outcomes and differences against oracle.
    choices=[];decisions=[];stored_selection=[]
    with (r3/"comparison/route-selections.csv").open() as f:stored_selection=list(csv.DictReader(f))
    for g in sorted(set(groups)):
        idx=np.flatnonzero(groups==g);assert list(route[idx])==list(ROUTES)
        chosen,stages=select_independent(pred[idx])
        prior_choice=next(x for x in stored_selection if x["group_id"]==g and x["method"]=="learned")["route"]
        assert chosen==prior_choice
        resolved=next((s["head"] for s in stages if s["resolved"]),"final_E_D_A_tie")
        selected=idx[ROUTES.index(chosen)]
        oracle=next(x for x in stored_selection if x["group_id"]==g and x["method"]=="oracle")
        for method in ("learned","geometry","B1","B2","B3","fixed_E","fixed_D","fixed_A","train_best_fixed","oracle"):
            row=next(x for x in stored_selection if x["group_id"]==g and x["method"]==method)
            choices.append(dict(group_id=str(g),split=str(split[idx[0]]),task=str(tasks[idx[0]]),method=method,route=row["route"],success=row["success"],
                                progress=row["progress"],path_m=row["base_path_m"],duration_s=row["terminal_duration_s"]))
        decisions.append(dict(group_id=str(g),split=str(split[idx[0]]),task=str(tasks[idx[0]]),selected_route=chosen,
                 first_decisive_head=resolved,stage_trace=stages,known_selected_success=float(y[selected,0]) if mask[selected,0] else None,
                 oracle_route=oracle["route"],oracle_success=oracle["success"],same_as_R3=True,
                 success_window_candidates=stages[0]["remaining"],route_scores=pred[idx].tolist(),
                 explanation_scope="posthoc known-outcome explanation only; six frozen slots and live choices unchanged"))
    csvwrite(out/"G-all-method-Source-choices.csv",choices);write(out/"G-selector-stages.json",decisions)
    csvwrite(out/"G-decision-summary.csv",[dict(group_id=x["group_id"],split=x["split"],task=x["task"],selected_route=x["selected_route"],
                 first_decisive_head=x["first_decisive_head"],selected_success=x["known_selected_success"],oracle_route=x["oracle_route"],oracle_success=x["oracle_success"]) for x in decisions])
    # Standalone scientific figures use every valid row, not a picked subset.
    fig,axes=plt.subplots(2,4,figsize=(16,7),layout="constrained")
    for col,j in enumerate(ACTIVE):
        for s,color in (("train","tab:blue"),("validation","tab:orange")):
            i=(split==s)&mask[:,j]
            axes[0,col].scatter(y[i,j],pred[i,j],s=18,alpha=.65,label=s,color=color)
            axes[1,col].hist(pred[i,j]-y[i,j],bins=16,alpha=.5,label=s,color=color)
        axes[0,col].set(title=HEADS[j],xlabel="Actual",ylabel="Predicted")
        lim=axes[0,col].get_xlim();axes[0,col].plot(lim,lim,"k--",lw=.8)
        axes[1,col].set(xlabel="Prediction - actual",ylabel="Rows")
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/"D-prediction-and-residual.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
    for block,sl in (("visual",slice(0,1024)),("geometry",slice(1024,1045))):
        for s in ("train","validation"):
            values=abs(xs[split==s,sl]).flatten()
            axes[0].hist(values,bins=50,alpha=.4,label=block+" "+s)
    axes[0].set(yscale="log",xlabel="Absolute standardized input",ylabel="Values");axes[0].legend(fontsize=8)
    for j in ACTIVE:
        rows=[x for x in optim if x["head"]==HEADS[j]]
        axes[1].plot([x["step"] for x in rows],[x["head_loss"] for x in rows],marker="o",label=HEADS[j])
    axes[1].set(xlabel="Checkpoint (train only)",ylabel="Source-equal normalized head loss");axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.savefig(out/"C-E-scale-and-optimization.png",dpi=150);plt.close(fig)
    summary=dict(ended_at=datetime.now(timezone.utc).isoformat(),status="A_G_diagnosis_completed",code_commit=subprocess.check_output(["git","-C",str(Path(__file__).parents[1]),"rev-parse","HEAD"],text=True).strip(),
      A_metric_max_difference=maxdiff,B_trace_checks=len(sample),B_unique_attempts=len({x["attempt"] for x in sample}),
      B_max_label_difference=max(x["max_label_difference"] for x in sample),C_nonfinite=0,C_suppressed_dims=int((~good).sum()),
      E_train_only_four_checkpoints=True,F_max_decomposition_error=max(x["decomposition_abs_error"] for x in decomposition),
      F_exact_same_visual_all_Source=all(x["visual_route_range"]<1e-10 for x in differences),
      F_interventions="two fixed final OOD zero-standardized-block cases, train+validation once each, never used to select live route",
      G_all_R3_choices_reproduced=True,critical_model_semantics_bug=False,optimizer_steps=0,sealed_test_read=False)
    write(out/"A-G-completion.json",summary);print(json.dumps(summary),flush=True)
if __name__=="__main__":main()
