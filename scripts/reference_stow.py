"""Shared, source-bound D stow target for planning and feedback control."""
import hashlib
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

VERSION = 'bounded-stow-fk-v1'
SOLVER_MARGIN = .05


def source_digest(model, data):
    return hashlib.sha256(np.asarray(data.qpos, dtype='<f8').tobytes()).hexdigest()


def build_stow(ref):
    model, live = ref.model_data()
    data = mujoco.MjData(model)
    data.qpos[:] = live.qpos
    data.mocap_pos[:] = live.mocap_pos
    data.mocap_quat[:] = live.mocap_quat
    mujoco.mj_forward(model, data)
    arm = ref.robot.part_controllers['right']
    site = ref.robot.eef_site_id['right']
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ref.base_body)
    qids = np.asarray(arm.qpos_index)
    joints = [int(np.flatnonzero(model.jnt_qposadr == q)[0]) for q in qids]
    limits = model.jnt_range[joints]
    offset = data.site_xpos[site] - data.xpos[body]
    offset[:2] *= max(0., 1. - .25 / max(np.linalg.norm(offset[:2]), 1e-6))
    preferred = dict(pos=data.xpos[body] + offset, rot=data.site_xmat[site].reshape(3, 3).copy())
    lower,upper=limits[:,0]+SOLVER_MARGIN,limits[:,1]-SOLVER_MARGIN
    def residual(q):
        data.qpos[qids]=q
        mujoco.mj_forward(model,data)
        ep=preferred['pos']-data.site_xpos[site]
        er=Rotation.from_matrix(preferred['rot']@data.site_xmat[site].reshape(3,3).T).as_rotvec()
        return np.r_[ep,.2*er]
    solution=least_squares(residual,np.clip(live.qpos[qids],lower,upper),
        bounds=(lower,upper),max_nfev=100,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    q=solution.x
    error=residual(q)
    pe=float(np.linalg.norm(error[:3]));re=float(np.linalg.norm(error[3:])/.2)
    # A bounded solution must remain close to the fixed preferred target.
    if not np.isfinite(q).all() or pe >= .012 or re >= .10:
        raise ValueError(f'bounded stow cannot meet preferred pose tolerance: position={pe}, rotation={re}')
    rotation = data.xmat[body].reshape(3, 3)
    return dict(version=VERSION, source_qpos_sha256=source_digest(model, live),
        arm_limits=limits.tolist(),
        solver_margin_rad=SOLVER_MARGIN, preferred_error_m=pe, preferred_error_rad=re,
        solver='scipy least_squares bounded TRF',solver_nfev=int(solution.nfev),
        local_pos=(rotation.T @ (data.site_xpos[site] - data.xpos[body])).tolist(),
        local_rot=(rotation.T @ data.site_xmat[site].reshape(3, 3)).tolist(), arm_qpos=q.tolist())


def load_stow(ref, record):
    model, data = ref.model_data()
    if record['version'] != VERSION or record['source_qpos_sha256'] != source_digest(model, data):
        raise ValueError('stow version or source mismatch')
    pos, rot, q = (np.asarray(record[k], dtype=float) for k in ('local_pos', 'local_rot', 'arm_qpos'))
    if pos.shape != (3,) or rot.shape != (3, 3) or q.shape != (7,) or not all(np.isfinite(x).all() for x in (pos, rot, q)):
        raise ValueError('invalid stow target')
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(rot), 1.):
        raise ValueError('invalid stow rotation')
    arm=ref.robot.part_controllers['right']
    joints=[int(np.flatnonzero(model.jnt_qposadr==i)[0]) for i in arm.qpos_index]
    limits=model.jnt_range[joints]
    if (record['solver_margin_rad'] != SOLVER_MARGIN
            or not np.array_equal(record['arm_limits'], limits)
            or np.min(np.minimum(q-limits[:,0],limits[:,1]-q)) < SOLVER_MARGIN-1e-12):
        raise ValueError('stow joint bounds mismatch')
    scratch=mujoco.MjData(model)
    scratch.qpos[:]=data.qpos;scratch.qpos[arm.qpos_index]=q
    scratch.mocap_pos[:]=data.mocap_pos;scratch.mocap_quat[:]=data.mocap_quat
    mujoco.mj_forward(model,scratch)
    body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,ref.base_body)
    site=ref.robot.eef_site_id['right'];R=scratch.xmat[body].reshape(3,3)
    if (not np.allclose(pos,R.T@(scratch.site_xpos[site]-scratch.xpos[body]),atol=1e-9,rtol=0)
            or not np.allclose(rot,R.T@scratch.site_xmat[site].reshape(3,3),atol=1e-9,rtol=0)):
        raise ValueError('stow FK mismatch')
    return pos, rot, q
