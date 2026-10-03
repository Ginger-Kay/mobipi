"""One frozen 2000-update A800 run; validation only after final checkpoint."""
import argparse, json, os, subprocess, hashlib, random, socket, time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from mobiwam.dr_v04_r3_learning import *
from dr_v04_r3_prepare import write, load, sha, csvwrite, HEADS

def now():return datetime.now(timezone.utc).isoformat()

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--stage",choices=["freeze","train"],required=True)
    a=p.parse_args();r=a.run;rt=Path(__file__).resolve().parent.parent
    schema=load(r/"dataset/schema.json")
    assert sha(r/"dataset/supervision.npz")==schema["dataset_sha256"]
    z=np.load(r/"dataset/supervision.npz",allow_pickle=False)
    train=z["split"]=="train";assert sum(train)==72
    X=z["X"][train];y=z["y"][train];mask=z["mask"][train];groups=z["group_id"][train]
    heads=support(y,mask,groups);active=[j for j,h in enumerate(heads) if h["status"]=="learnable"]
    participating=set(groups[mask[:,active].any(1)]) if active else set()
    scaleidx=np.array([g in participating for g in groups])
    mean,std=fit_scaler(X[scaleidx],groups[scaleidx]) if active else (np.zeros(1045),np.ones(1045))
    binding=dict(created_at=now(),training_seed=17,structure="Linear(1045,5)",structural_parameters=5230,
        trainable_parameters=len(active)*1046,active_heads=active,head_support=heads,source_count=24,
        optimization_sources=len(participating),scaler_sources=sorted(participating),scaler_rows=int(scaleidx.sum()),
        scaler_semantics="all three preoutcome routes within participating train Sources; population std; std<1e-6 output zero",
        mean=mean.tolist(),std=std.tolist(),full_batch_rows=72,masked_effective_rows=int(mask.any(1).sum()),
        optimizer="AdamW",lr=3e-4,betas=[.9,.999],eps=1e-8,amsgrad=False,weight_decay=.05,bias_decay=0.,
        grad_clip_global_L2=1.,steps=2000,epochs=2000,scheduler="CosineAnnealingLR",T_max=2000,eta_min=1e-5,
        scheduler_order="after optimizer.step",smooth_l1_beta=.1,BCE_pos_weight=None,
        precision="float32",AMP=False,torch_cpu_threads=4,checkpoint_steps=[500,1000,1500,2000],
        checkpoint_selection="fixed_step2000",validation_access_before_final_checkpoint=False,
        source_loss="per head within Source valid-route average, then labeled-Source average, then learnable-head average",
        dataset_sha256=schema["dataset_sha256"],dataset=str(r/"dataset/supervision.npz"),schema=schema,
        code_commit=subprocess.check_output(["git","-C",str(rt),"rev-parse","HEAD"],text=True).strip(),
        research_commit="1cd5e91cacd68060fedc62203265e15f71bab1e8",torch_version=torch.__version__,
        deterministic_algorithms=True,cudnn_benchmark=False,tf32=False,CUBLAS_WORKSPACE_CONFIG=":4096:8",
        selected_device="A800 CUDA; physical mapping in preflight/gpu-selection.json",
        initialization="torch.nn.Linear default, complete 5-head initialization with seed17; active rows copied",
        formal_train_ready=False,sealed_test_read=False)
    cfg=r/"training/config.json"
    if a.stage=="freeze":
        assert not cfg.exists();write(cfg,binding);cfg.chmod(0o444)
        print(json.dumps({k:binding[k] for k in ("head_support","optimization_sources","trainable_parameters","scaler_rows")}),flush=True);return
    frozen=load(cfg)
    assert frozen["code_commit"]==binding["code_commit"] and frozen["dataset_sha256"]==binding["dataset_sha256"]
    for key in ("mean","std","head_support","active_heads"):assert frozen[key]==binding[key]
    assert not (r/"training/result.json").exists()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(17);np.random.seed(17);random.seed(17)
    assert torch.cuda.is_available() and "A800" in torch.cuda.get_device_name(0)
    device=torch.device("cuda:0")
    model=ActiveLinear(active).to(device)
    x=torch.from_numpy(scale(X,mean,std)).to(device)
    targets=torch.from_numpy((y/SCALES).astype(np.float32)).to(device)
    valid=torch.from_numpy(mask).to(device)
    _,gids=np.unique(groups,return_inverse=True);g=torch.from_numpy(gids).to(device)
    initial=model.materialized()
    write(r/"training/process.json",dict(started_at=now(),pid=os.getpid(),host=socket.gethostname(),command=subprocess.list2cmdline(__import__("sys").argv),
          sys_executable=__import__("sys").executable,device=str(device),device_name=torch.cuda.get_device_name(0),
          CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),torch_module=torch.__file__,model_device=str(model.weight.device),
          batch_device=str(x.device),config_sha256=sha(cfg)))
    if not active:
        write(r/"training/result.json",dict(status="no_learnable_head",optimizer_steps=0));return
    opt=torch.optim.AdamW([dict(params=[model.weight],weight_decay=.05),dict(params=[model.bias],weight_decay=0.)],
                         lr=3e-4,betas=(.9,.999),eps=1e-8,amsgrad=False)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=2000,eta_min=1e-5)
    history=[]
    with (r/"training/loss.jsonl").open("x") as log:
        initial_loss,h=masked_loss(model(x),targets,valid,g,active)
        record=dict(step=0,at=now(),loss=float(initial_loss.detach()),head_loss={HEADS[j]:float(v.detach()) for j,v in h.items()},lr=opt.param_groups[0]["lr"],gradient_norm=None)
        log.write(json.dumps(record)+"\n");log.flush();print(json.dumps(record),flush=True)
        for step in range(1,2001):
            opt.zero_grad(set_to_none=True)
            loss,h=masked_loss(model(x),targets,valid,g,active)
            if not torch.isfinite(loss):raise FloatingPointError("nonfinite loss")
            loss.backward();norm=torch.nn.utils.clip_grad_norm_([model.weight,model.bias],1.)
            if not torch.isfinite(norm):raise FloatingPointError("nonfinite gradient")
            opt.step();scheduler.step()
            if step%10==0:
                with torch.no_grad():post,ph=masked_loss(model(x),targets,valid,g,active)
                record=dict(step=step,at=now(),loss=float(post),head_loss={HEADS[j]:float(v) for j,v in ph.items()},
                            gradient_norm=float(norm),lr=opt.param_groups[0]["lr"])
                history.append(record);log.write(json.dumps(record)+"\n");log.flush();print(json.dumps(record),flush=True)
            if step in (500,1000,1500,2000):
                cp=r/"training"/f"step{step:04d}.pt"
                assert not cp.exists()
                torch.save(dict(step=step,model=model.state_dict(),linear=model.materialized(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
                           rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate()),
                           scaler=dict(mean=mean,std=std),head_support=heads,label_mask=mask,schema=schema,config=frozen,
                           config_sha256=sha(cfg),dataset_sha256=schema["dataset_sha256"],code_commit=frozen["code_commit"]),cp)
                if (r/"training/latest").is_symlink():(r/"training/latest").unlink()
                (r/"training/latest").symlink_to(cp.name)
    final=model.materialized()
    recent=[h for h in history if h["step"]>=1800]
    delta=recent[0]["loss"]-recent[-1]["loss"]
    write(r/"training/result.json",dict(status="completed_development_only",ended_at=now(),optimizer_steps=2000,
          final_checkpoint=str(r/"training/step2000.pt"),final_checkpoint_sha256=sha(r/"training/step2000.pt"),
          initial_train_loss=float(initial_loss.detach()),final_train_loss=history[-1]["loss"],last200_loss_absolute_drop=delta,
          last200_loss_relative_drop=delta/max(abs(recent[0]["loss"]),1e-12),
          optimization_not_settled=delta/max(abs(recent[0]["loss"]),1e-12)>.01,
          final_weight_norm=float(torch.linalg.vector_norm(final["weight"])),
          parameter_change_norm=float(torch.linalg.vector_norm(torch.cat([(final["weight"]-initial["weight"]).flatten(),final["bias"]-initial["bias"]]))),
          checkpoint_count=4,trainable_parameters=frozen["trainable_parameters"],device="cuda:0",
          unsupported_or_constant_heads=[HEADS[j] for j in range(5) if j not in active],formal_train_ready=False))
    # Final checkpoint is frozen before any validation prediction.
    from dr_v04_r3_compare import evaluate_final
    evaluate_final(r)

if __name__=="__main__":main()
