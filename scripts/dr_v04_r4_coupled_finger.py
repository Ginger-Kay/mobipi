"""Use the real Panda 1-D closing-command signs for a scratch geometry scan."""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mobiwam.task_video_identity import source_model,sha
from mobiwam.reference_collision import SweptGeometry

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();r=a.run
    previous=json.loads((r/"diagnosis/H-saved-finger-evidence.json").read_text())
    item=previous["receipts"][0]
    roster=json.loads((r/"frozen-six/roster.json").read_text());r3=Path(roster["r3_root"])
    original=json.loads((r3/"audit/remaining-pair-rejections.json").read_text())["records"][0]
    path=Path(json.loads(Path(original["original_receipt"]).read_text())["attempt"])
    xml=path.parent.parent/"model.xml";m=source_model(str(xml),sha(xml));g=SweptGeometry(m,target_prefix="")
    with np.load(path/"formal-native-substeps.npz") as z:q=z["qpos"][item["saved_absolute_segment"]+1].copy()
    joints=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,"gripper0_right_finger_joint"+str(k)) for k in (1,2)]
    assert np.allclose(m.jnt_range[joints],[[0,.04],[-.04,0]])
    controller=Path("/share/personal/chensiyu/haokaijiang/MobiWAM/env/lib/python3.10/site-packages/robosuite/models/grippers/panda_gripper.py")
    assert "np.array([-1.0, 1.0])" in controller.read_text()
    pairs=[["gripper0_right_finger1_collision","gripper0_right_finger2_pad_collision"],
           ["gripper0_right_finger1_pad_collision","gripper0_right_finger2_collision"],
           ["gripper0_right_finger1_pad_collision","gripper0_right_finger2_pad_collision"]]
    values=[]
    for u in np.linspace(0,1,81):
        pose=q.copy();pose[m.jnt_qposadr[joints[0]]]=.04*(1-u);pose[m.jnt_qposadr[joints[1]]]=-.04*(1-u)
        g.d.qpos[:]=pose;mujoco.mj_forward(g.m,g.d)
        for pair in pairs:
            ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n) for n in pair]
            values.append(dict(closing_fraction=float(u),q_finger1=float(.04*(1-u)),q_finger2=float(-.04*(1-u)),
                               pair=pair,distance_m=float(g.geom_distance(*ids))))
    out=dict(created_at=datetime.now(timezone.utc).isoformat(),schema="PandaGripper-real-coupled-kinematic-closure-v1",
             controller=str(controller),controller_sha256=sha(controller),format_action_closed_signs=[-1,1],
             joint_ranges=m.jnt_range[joints].tolist(),fully_open_qpos=[.04,-.04],fully_closed_command_qpos=[0,0],
             replaced_interpretation="H-joint-range-distance.png scans both independent joint ranges in the same fraction direction; it is not the normal coupled closing path. This file supplies the real opposite-sign coupled path.",
             values=values,physics_steps=0,actual_dynamic_reachability_claim=False,monitor_not_modified=True)
    (r/"diagnosis/H-coupled-closure.json").write_text(json.dumps(out,indent=2))
    fig,ax=plt.subplots(figsize=(9,4),layout="constrained")
    for pair in pairs:
        v=[x for x in values if x["pair"]==pair]
        ax.plot([x["closing_fraction"] for x in v],[x["distance_m"]*1000 for x in v],label=" / ".join(n.replace("gripper0_right_","") for n in pair))
    ax.axhline(.5,color="k",ls="--",label="Retained 0.5mm monitor");ax.axhline(0,color="k",lw=.7)
    ax.set(xlabel="Panda coupled closing fraction (open 0 -> closed-command 1)",ylabel="Signed distance (mm)",title="Scratch geometry from actual actuator directions, no dynamics")
    ax.legend(fontsize=7);ax.grid(alpha=.2);fig.savefig(r/"diagnosis/H-coupled-closure.png",dpi=160);plt.close(fig)
    print(json.dumps(dict(status="coupled_closure_completed",values=len(values),
      minima=[dict(pair=p,distance_m=min(x["distance_m"] for x in values if x["pair"]==p)) for p in pairs])),flush=True)
if __name__=="__main__":main()
