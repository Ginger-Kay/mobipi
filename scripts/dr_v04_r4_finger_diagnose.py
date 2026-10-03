"""H: saved states only, native physical filters and gripper geometry."""
import argparse,json,csv
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,mujoco
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mobiwam.task_video_identity import source_model,sha
from mobiwam.reference_collision import SweptGeometry
from mobiwam.contact_rules import FINGER_PAD_PAIR

def load(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+"\n")
def chain(m,b):
    result=[]
    while b:result.append(int(b));b=int(m.body_parentid[b])
    return result
def name(m,kind,i):return mujoco.mj_id2name(m,kind,int(i)) or ""
def details(m,g):
    body=int(m.geom_bodyid[g]);anc=chain(m,body);joints=[]
    for b in anc:
        for j in range(int(m.body_jntadr[b]),int(m.body_jntadr[b]+m.body_jntnum[b])):
            joints.append(dict(id=j,name=name(m,mujoco.mjtObj.mjOBJ_JOINT,j),type=int(m.jnt_type[j]),qpos_address=int(m.jnt_qposadr[j]),range=m.jnt_range[j].tolist(),
                               axis=m.jnt_axis[j].tolist(),limited=bool(m.jnt_limited[j])))
    return dict(geom_id=g,geom_name=name(m,mujoco.mjtObj.mjOBJ_GEOM,g),body_id=body,body_name=name(m,mujoco.mjtObj.mjOBJ_BODY,body),
        weld_id=int(m.body_weldid[body]),parent_id=int(m.body_parentid[body]),ancestors=anc,ancestral_joints=joints,
        geom_type=int(m.geom_type[g]),contype=int(m.geom_contype[g]),conaffinity=int(m.geom_conaffinity[g]),local_pos=m.geom_pos[g].tolist())
def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();r=a.run
    r3=Path(load(r/"frozen-six/roster.json")["r3_root"])
    rejected=load(r3/"audit/remaining-pair-rejections.json")["records"];receipts=[];ranges=[]
    names4=["gripper0_right_finger1_collision","gripper0_right_finger1_pad_collision","gripper0_right_finger2_collision","gripper0_right_finger2_pad_collision"]
    for k,item in enumerate(rejected):
        old=load(item["original_receipt"]);path=Path(old["attempt"]);xml=path.parent.parent/"model.xml"
        model=source_model(str(xml),sha(xml));geom=SweptGeometry(model,target_prefix=load(path/"task-video-manifest.json")["binding"]["native"]["fixture_name"])
        with np.load(path/"formal-native-substeps.npz",allow_pickle=False) as z:
            index=item["absolute_segment"];qpos=z["qpos"][index+1].copy();last=z["qpos"][-1].copy();sim_time=float(z["sim_time"][index+1])
        ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in item["first_remaining_pair"]]
        info=[details(model,g) for g in ids];ba,bb=[x["body_id"] for x in info]
        common=next(b for b in chain(model,ba) if b in chain(model,bb))
        joints=sorted({j["id"] for x in info for j in x["ancestral_joints"] if "gripper0_right_finger_joint" in j["name"]})
        assert len(joints)==2
        pair_in_monitor=any(set(pair)==set(ids) for pair in geom.pairs)
        compatible=bool((model.geom_contype[ids[0]]&model.geom_conaffinity[ids[1]]) or (model.geom_contype[ids[1]]&model.geom_conaffinity[ids[0]]))
        phys=mujoco.MjData(model);phys.qpos[:]=qpos;mujoco.mj_forward(model,phys)
        native_contacts=[dict(geom1=name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),geom2=name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom2),distance_m=float(c.dist))
                         for c in phys.contact[:phys.ncon] if set((int(c.geom1),int(c.geom2)))==set(ids)]
        geom.d.qpos[:]=qpos;mujoco.mj_forward(geom.m,geom.d)
        signed=geom.geom_distance(*ids)
        phys.qpos[:]=last;mujoco.mj_forward(model,phys)
        terminal_contacts=[dict(geom1=name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),geom2=name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom2),distance_m=float(c.dist))
                  for c in phys.contact[:phys.ncon] if "gripper0_right_finger" in name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom1) and
                     "gripper0_right_finger" in name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)]
        rec=dict(group_id=item["group_id"],route=item["route"],saved_absolute_segment=index,saved_sim_time=sim_time,pair=item["first_remaining_pair"],geometries=info,
         same_body=ba==bb,same_weld=info[0]["weld_id"]==info[1]["weld_id"],same_direct_parent=info[0]["parent_id"]==info[1]["parent_id"],
         common_ancestor=name(model,mujoco.mjtObj.mjOBJ_BODY,common),physical_filter_compatible=compatible,
         retained_in_R3_monitor=pair_in_monitor,explicit_model_pairs=int(model.npair),counterexample_signed_distance_m=float(signed),
         actual_native_contact_at_counterexample=bool(native_contacts),native_contacts=native_contacts,
         final_saved_native_finger_contacts=terminal_contacts,
         saved_finger_qpos={name(model,mujoco.mjtObj.mjOBJ_JOINT,j):float(qpos[model.jnt_qposadr[j]]) for j in joints},
         saved_state_readonly=True,env_step_calls=0,normal_motion_scope="both finger joints within original physical joint limits; legality of monitoring exemption remains researcher decision")
        receipts.append(rec)
        if k==0:
            # Paired finger motion inside native joint bounds, on scratch data only.
            for t in np.linspace(0,1,41):
                q=qpos.copy()
                for j in joints:q[model.jnt_qposadr[j]]=(1-t)*model.jnt_range[j,0]+t*model.jnt_range[j,1]
                geom.d.qpos[:]=q;mujoco.mj_forward(geom.m,geom.d)
                for n1,n2 in [names4[:1]+names4[3:],names4[1:3],sorted(FINGER_PAD_PAIR)]:
                    gg=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in (n1,n2)]
                    ranges.append(dict(fraction_of_native_joint_range=float(t),joint_positions=[float(q[model.jnt_qposadr[j]]) for j in joints],
                                  pair=[n1,n2],signed_distance_m=float(geom.geom_distance(*gg)),scratch_geometry_only=True))
            geom.d.qpos[:]=qpos;mujoco.mj_forward(geom.m,geom.d)
            hand=common;center=geom.d.xpos[hand];rotation=geom.d.xmat[hand].reshape(3,3)
            fig,ax=plt.subplots(figsize=(8,5),layout="constrained")
            for n,color in zip(names4,["tab:blue","tab:cyan","tab:red","tab:orange"]):
                g=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,n);typ=model.geom_type[g]
                if typ==mujoco.mjtGeom.mjGEOM_MESH:
                    mesh=model.geom_dataid[g];vertices=model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
                elif typ==mujoco.mjtGeom.mjGEOM_BOX:
                    vertices=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*model.geom_size[g]
                else:continue
                world=vertices@geom.d.geom_xmat[g].reshape(3,3).T+geom.d.geom_xpos[g]
                local=(world-center)@rotation
                projection=local[:,[1,2]]*1000
                hull=ConvexHull(projection)
                poly=projection[hull.vertices];ax.fill(poly[:,0],poly[:,1],alpha=.25,color=color,label=n.replace("gripper0_right_",""))
                ax.plot(np.r_[poly[:,0],poly[0,0]],np.r_[poly[:,1],poly[0,1]],color=color,lw=1)
            ax.set(xlabel="Hand-local Y (mm)",ylabel="Hand-local Z (mm)",title="Saved finger geometry: 2D projection, not a penetration certificate")
            ax.axis("equal");ax.legend(fontsize=8);ax.grid(alpha=.2);fig.savefig(r/"diagnosis/H-finger-geometry.png",dpi=160);plt.close(fig)
            fig,ax=plt.subplots(figsize=(8,4),layout="constrained")
            for pair in {tuple(x["pair"]) for x in ranges}:
                vals=[x for x in ranges if tuple(x["pair"])==pair]
                ax.plot([x["fraction_of_native_joint_range"] for x in vals],[x["signed_distance_m"]*1000 for x in vals],label=" / ".join(n.replace("gripper0_right_","") for n in pair))
            ax.axhline(.5,color="k",ls="--",label="Retained monitor 0.5mm");ax.axhline(0,color="k",lw=.7)
            ax.set(xlabel="Paired finger position within native limits",ylabel="Signed geometry distance (mm)",title="Scratch kinematics only: no physics stepping")
            ax.legend(fontsize=7);ax.grid(alpha=.2);fig.savefig(r/"diagnosis/H-joint-range-distance.png",dpi=160);plt.close(fig)
        print(json.dumps(dict(group=item["group_id"],route=item["route"],distance=signed,monitor=pair_in_monitor,same_weld=rec["same_weld"])),flush=True)
        source_model.cache_clear()
    write(r/"diagnosis/H-saved-finger-evidence.json",dict(created_at=datetime.now(timezone.utc).isoformat(),status="H_completed_readonly",
      receipts=receipts,joint_range_measurements=ranges,physics_modified=False,monitor_exemption_expanded=False,
      interpretation="finger siblings share hand ancestor but different articulated/weld bodies; physical pair compatibility and retained monitoring agree; first counterexamples are positive low clearance, not native penetration",
      recommendation="Research can decide whether exact sibling finger-entity/opposite-pad pairs should join the monitoring exemption; retain native contact response and all other robot/environment pairs. No change in this run."))
if __name__=="__main__":main()
