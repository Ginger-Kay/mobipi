"""V3 live kinematics, actuator mapping, and predictive clearance constraints.

All geometry is simulator-oracle pre-outcome input. This module does not change
the simulator model, task checker, frozen policy, or action-query boundaries.
"""
from __future__ import annotations

import numpy as np
import mujoco

from mobiwam.mobipi_actions import matrix_to_axis_angle
from mobiwam.planner_min import velocity_level_qp


class LiveControl:
    def __init__(self, adapter):
        self.adapter = adapter
        self.raw = adapter._unwrapped()
        self.sim = self.raw.sim
        self.model = self.sim.model._model
        self.data = self.sim.data._data
        self.scratch = mujoco.MjData(self.model)
        self.robot = self.raw.robots[0]
        self.arm = self.robot.part_controllers["right"]
        self.base = self.robot.part_controllers["base"]
        self.arm_dofs = np.asarray(self.arm.qvel_index, int)
        self.base_dofs = np.asarray(self.base.qvel_index, int)
        self.dofs = np.r_[self.base_dofs, self.arm_dofs]
        self.qpos_indices = np.r_[self.base.qpos_index, self.arm.qpos_index].astype(int)
        self.site = int(self.robot.eef_site_id["right"])
        self.context = adapter._live_articulation_context()
        self.dt = 1 / self.raw.control_freq
        self.names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(self.model.ngeom)]
        collision_geoms = [i for i in range(self.model.ngeom) if self.model.geom_contype[i] or self.model.geom_conaffinity[i]]
        robot_geoms = [i for i in collision_geoms if self.names[i].startswith(("robot0", "mobilebase0", "gripper0"))]
        world_geoms = [i for i in collision_geoms if i not in robot_geoms and "floor" not in self.names[i]]
        self.pairs = []
        handles = set(self.context["handle_geom_ids"])
        fingers = set(self.context["robot_handle_geom_ids"])
        for g in robot_geoms:
            for h in world_geoms:
                if g in fingers and h in handles:
                    continue
                if not (self.model.geom_contype[g] & self.model.geom_conaffinity[h] or self.model.geom_contype[h] & self.model.geom_conaffinity[g]):
                    continue
                self.pairs.append((g, h, .05 if self.names[g].startswith("mobilebase0") else .002))
        # Ignore same-link / neighboring-link structural geometry. Nonadjacent
        # arm self collision remains constrained; mobilebase support duplicates
        # are structural and intentionally overlap in the source XML.
        for ix, g in enumerate(robot_geoms):
            for h in robot_geoms[ix+1:]:
                bg, bh = int(self.model.geom_bodyid[g]), int(self.model.geom_bodyid[h])
                ancestors_g = {bg, int(self.model.body_parentid[bg]), int(self.model.body_parentid[self.model.body_parentid[bg]])}
                ancestors_h = {bh, int(self.model.body_parentid[bh]), int(self.model.body_parentid[self.model.body_parentid[bh]])}
                if ancestors_g & ancestors_h or (self.names[g].startswith("mobilebase0") and self.names[h].startswith("mobilebase0")):
                    continue
                if not (self.model.geom_contype[g] & self.model.geom_conaffinity[h] or self.model.geom_contype[h] & self.model.geom_conaffinity[g]):
                    continue
                self.pairs.append((g,h,.002))

    def jacobian(self, data=None):
        data = self.data if data is None else data
        jp, jr = np.zeros((3,self.model.nv)), np.zeros((3,self.model.nv))
        mujoco.mj_jacSite(self.model,data,jp,jr,self.site)
        return np.vstack([jp[:,self.dofs],jr[:,self.dofs]])

    def clearance(self, data=None, gradients=False):
        data = self.data if data is None else data
        result=[]
        for g,h,margin in self.pairs:
            # Bounding spheres reject far pairs without changing exact near checks.
            if np.linalg.norm(data.geom_xpos[g]-data.geom_xpos[h]) > self.model.geom_rbound[g]+self.model.geom_rbound[h]+.12:
                continue
            segment=np.zeros(6)
            distance=float(mujoco.mj_geomDistance(self.model,data,g,h,.12,segment))
            if distance >= .12:
                continue
            row={"pair":[self.names[g],self.names[h]], "distance_m":distance, "margin_m":margin,
                 "signed_clearance_m":distance-margin}
            if gradients:
                direction=segment[3:]-segment[:3]
                direction /= max(np.linalg.norm(direction),1e-12)
                jg,jh=np.zeros((3,self.model.nv)),np.zeros((3,self.model.nv))
                mujoco.mj_jac(self.model,data,jg,None,segment[:3],int(self.model.geom_bodyid[g]))
                mujoco.mj_jac(self.model,data,jh,None,segment[3:],int(self.model.geom_bodyid[h]))
                row["gradient"]=direction@(jh-jg)[:,self.dofs]
                row["fixture_gradient"]=float(direction@(jh-jg)[:,int(self.model.jnt_dofadr[self.context["joint_id"]])])
            result.append(row)
        return sorted(result,key=lambda r:(r["signed_clearance_m"],r["pair"]))

    def allocate(self, twist, base_reference, previous_velocity, fixture_qvel=0):
        jac=self.jacobian()
        # A base support objective is part of the same least-squares QP.
        augmented=np.vstack([jac, np.c_[np.eye(3)*.4,np.zeros((3,7))]])
        target=np.r_[twist,np.asarray(base_reference)*.4]
        lo=np.r_[[-.2,-.2,-.2],np.full(7,-.35)]
        hi=-lo
        q=self.data.qpos[self.qpos_indices]
        for k,index in enumerate(self.qpos_indices):
            jid=int(np.flatnonzero(self.model.jnt_qposadr==index)[0])
            if self.model.jnt_limited[jid]:
                lo[k]=max(lo[k],(self.model.jnt_range[jid,0]+.015-q[k])/self.dt)
                hi[k]=min(hi[k],(self.model.jnt_range[jid,1]-.015-q[k])/self.dt)
        acceleration=np.r_[np.full(3,.4),np.full(7,1.0)]
        lo=np.maximum(lo,np.asarray(previous_velocity)-acceleration*self.dt)
        hi=np.minimum(hi,np.asarray(previous_velocity)+acceleration*self.dt)
        near=self.clearance(gradients=True)
        matrix=np.asarray([r["gradient"] for r in near]).reshape(-1,10)
        bound=np.asarray([(-r["signed_clearance_m"])/self.dt-r["fixture_gradient"]*fixture_qvel for r in near])
        velocity,receipt=velocity_level_qp(augmented,target,lo,hi,base_weight=.25,damping=.003,
                                         inequalities=(matrix,bound),projection_iterations=96)
        receipt["constraint_min_residual"]=float(np.min(matrix@velocity-bound)) if len(bound) else None
        receipt["nearest"]=[{k:v for k,v in r.items() if k not in ("gradient",)} for r in near[:3]]
        receipt["jacobian_rank"]=int(np.linalg.matrix_rank(jac))
        receipt["base_dofs"]=self.base_dofs.tolist();receipt["arm_dofs"]=self.arm_dofs.tolist()
        return velocity,receipt

    def mapped_action(self, velocity, nominal, scale=1.0):
        # Apply common time scale to generalized base+arm velocity before the
        # nonlinear friction compensation. Never multiply only the base afterward.
        velocity=np.asarray(velocity)*scale
        action=np.asarray(nominal,float).copy()
        jac=self.jacobian()
        arm_twist=jac[:,3:]@velocity[3:]
        rotation=np.asarray(self.arm.origin_ori)
        action[:3]=(rotation.T@arm_twist[:3])*self.dt/np.asarray(self.arm.output_max)[:3]
        action[3:6]=(rotation.T@arm_twist[3:])*self.dt/np.asarray(self.arm.output_max)[3:6]
        # Coulomb feed-forward is derived from the frozen XML frictionloss and
        # actuator gain, then transformed through the live controller mapping.
        indices=np.asarray(self.robot._ref_actuators_indexes_dict["base"],int)
        gain=self.model.actuator_gainprm[indices,0]
        weight=.5*(self.base.actuator_max-self.base.actuator_min)
        requested=velocity[:3].copy()
        friction=self.model.dof_frictionloss[self.base_dofs]/gain
        requested += np.where(np.abs(requested)>1e-6,np.sign(requested)*friction,0)
        goal=requested/weight
        _,orientation=self.base.get_base_pose()
        current=np.arctan2(orientation[1,0],orientation[0,0])
        initial=np.arctan2(self.base.init_ori[1,0],self.base.init_ori[0,0])
        theta=current-initial
        mapping=np.array([[-np.sin(theta),np.cos(theta),0],[np.cos(theta),np.sin(theta),0],[0,0,1]])
        action[7:10]=np.linalg.solve(mapping,goal)
        action[11]=-1.0  # achieved-pose deltas, explicitly bound by mapping probe
        return action

    def swept(self, velocity, fixture_qvel=0):
        self.scratch.qpos[:]=self.data.qpos
        self.scratch.qvel[:]=self.data.qvel
        full=np.zeros(self.model.nv);full[self.dofs]=velocity
        full[int(self.model.jnt_dofadr[self.context["joint_id"]])]=fixture_qvel
        nearest=None
        for fraction in (.25,.5,.75,1.0):
            self.scratch.qpos[:]=self.data.qpos
            mujoco.mj_integratePos(self.model,self.scratch.qpos,full,self.dt*fraction)
            mujoco.mj_forward(self.model,self.scratch)
            rows=self.clearance(self.scratch)
            if rows and (nearest is None or rows[0]["signed_clearance_m"]<nearest["signed_clearance_m"]):
                nearest={**rows[0],"fraction":fraction}
        return {"passed":nearest is None or nearest["signed_clearance_m"]>=-1e-7,"nearest":nearest}

    def handle_pose(self):
        result=np.eye(4)
        if self.context["handle_site_ids"]:
            index=self.context["handle_site_ids"][0]
            result[:3,3]=self.data.site_xpos[index];result[:3,:3]=self.data.site_xmat[index].reshape(3,3)
        else:
            index=self.context["handle_geom_ids"][0]
            result[:3,3]=self.data.geom_xpos[index];result[:3,:3]=self.data.geom_xmat[index].reshape(3,3)
        return result

    def orientation_error(self, target, current):
        return matrix_to_axis_angle(target@current.T)
