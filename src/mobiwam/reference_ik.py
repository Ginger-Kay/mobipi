"""Collision-constrained local pose IK for development geometric plans.

The environment and control law are not changed. Distance constraints use the
same pair set as SweptGeometry; acceptance still requires its swept validator.
"""
import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def distance_rows(check, q, phase, dofs, activation=.02):
    pairs, distances = check.distances(q, phase)
    active = np.flatnonzero(distances < activation)
    jac_a = np.zeros((3, check.m.nv)); jac_b = jac_a.copy()
    rotational = jac_a.copy(); rows = []
    for i in active:
        a, b = map(int, pairs[i]); segment = np.zeros(6)
        distance = mujoco.mj_geomDistance(check.m, check.d, a, b, .10, segment)
        normal = segment[3:] - segment[:3]
        length = np.linalg.norm(normal)
        if length < 1e-12:
            # A zero derivative cannot certify escape from coincident surfaces.
            rows.append(np.zeros(len(dofs))); continue
        normal /= length
        if distance < 0: normal = -normal
        mujoco.mj_jac(check.m, check.d, jac_a, rotational, segment[:3], int(check.m.geom_bodyid[a]))
        mujoco.mj_jac(check.m, check.d, jac_b, rotational, segment[3:], int(check.m.geom_bodyid[b]))
        rows.append(normal @ (jac_b[:, dofs] - jac_a[:, dofs]))
    return np.asarray(rows).reshape(-1, len(dofs)), distances[active]


def constrained_pose_ik(model, data, site, qids, dofs, target, seed, limits,
                        check, phase='manipulate', iterations=80, buffer=.001):
    """Sequential bounded quadratic steps; no relaxed collision acceptance.

    A 1mm solver buffer is stricter than the 0.5mm swept gate. Unreachable pose
    and infeasible constraints remain explicit failures, never safe fallbacks.
    """
    data.qpos[qids] = seed
    jp = np.zeros((3, model.nv)); jr = jp.copy()
    history = []; status = 'iteration_limit'
    for iteration in range(iterations):
        mujoco.mj_forward(model, data)
        ep = target['pos'] - data.site_xpos[site]
        er = Rotation.from_matrix(target['rot'] @ data.site_xmat[site].reshape(3, 3).T).as_rotvec()
        mujoco.mj_jacSite(model, data, jp, jr, site)
        J = np.vstack([jp[:, dofs], .2 * jr[:, dofs]])
        error = np.r_[ep, .2 * er]
        A, distances = distance_rows(check, data.qpos, phase, dofs)
        minimum = float(min(distances, default=.10))
        history.append((float(np.linalg.norm(ep)), float(np.linalg.norm(er)), minimum))
        if np.linalg.norm(ep) < .0002 and np.linalg.norm(er) < .002 and minimum >= buffer - 1e-8:
            status = 'converged'; break
        q = data.qpos[qids].copy()
        lower = np.maximum(-.08, limits[:, 0] + .01501 - q)
        upper = np.minimum(.08, limits[:, 1] - .01501 - q)
        if np.any(lower > upper): status = 'joint_bounds_infeasible'; break
        H = J.T @ J + np.eye(len(qids)) * .00002
        gradient = -J.T @ error
        constraints = []
        if len(A):
            rhs = buffer - distances
            constraints = [dict(type='ineq', fun=lambda x: A @ x - rhs, jac=lambda x: A)]
        step = minimize(lambda x: .5 * x @ H @ x + gradient @ x,
                        np.zeros(len(qids)), jac=lambda x: H @ x + gradient,
                        bounds=list(zip(lower, upper)), constraints=constraints,
                        method='SLSQP', options=dict(ftol=1e-12, maxiter=60))
        if not step.success:
            status = 'local_constraints_infeasible'; break
        data.qpos[qids] = q + step.x
        if np.linalg.norm(step.x) < 1e-7:
            status = 'constrained_stationary'; break
    mujoco.mj_forward(model, data)
    pe = float(np.linalg.norm(target['pos'] - data.site_xpos[site]))
    re = float(Rotation.from_matrix(target['rot'] @ data.site_xmat[site].reshape(3, 3).T).magnitude())
    _, distances = check.distances(data.qpos, phase)
    receipt = dict(status=status, iterations=len(history), position_error_m=pe,
                   rotation_error_rad=re, minimum_endpoint_distance_m=float(min(distances, default=.10)),
                   solver_buffer_m=buffer, collision_gate_relaxed=False)
    return data.qpos[qids].copy(), pe, re, receipt
