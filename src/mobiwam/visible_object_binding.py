"""Inspect actual native visible/collision geometry, not task/asset names.

The baseline is the compiled immutable Source. This detects a changed binding,
mesh, filter, local transform or stale cached ID, but does not automatically
recognize the semantic class of an unreviewed asset.
"""
import hashlib,json
import numpy as np,mujoco

class VisibleBindingHold(ValueError):
    pass

def name(model,kind,index):return mujoco.mj_id2name(model,kind,int(index)) or ''

def ancestry(model,body):
    result=[]
    while body:
        first=int(model.body_jntadr[body]);count=int(model.body_jntnum[body])
        result.append(dict(body_id=int(body),body_name=name(model,mujoco.mjtObj.mjOBJ_BODY,body),
            joints=[dict(joint_id=j,name=name(model,mujoco.mjtObj.mjOBJ_JOINT,j),qpos_address=int(model.jnt_qposadr[j]),type=int(model.jnt_type[j])) for j in range(first,first+count)]))
        body=int(model.body_parentid[body])
    return result

def model_geometry_sha(model):
    """Fingerprint static model geometry, including runtime mutable transforms.

    Physics qpos/qvel and free-camera controls are intentionally excluded.
    Compiled mesh vertices/faces supply shape identity independently of a name.
    """
    h=hashlib.sha256()
    for key in ('names','body_parentid','body_geomadr','body_geomnum','body_jntadr','body_jntnum','body_pos','body_quat',
                'jnt_bodyid','jnt_type','jnt_qposadr','jnt_axis','jnt_pos','jnt_range',
                'geom_type','geom_bodyid','geom_dataid','geom_size','geom_pos','geom_quat',
                'geom_group','geom_contype','geom_conaffinity','geom_rgba','geom_matid',
                'mesh_vertadr','mesh_vertnum','mesh_faceadr','mesh_facenum','mesh_vert','mesh_face'):
        value=getattr(model,key);h.update(key.encode());h.update(value if isinstance(value,bytes) else np.asarray(value).tobytes())
    return h.hexdigest()

def mesh_shape_sha(model,mesh):
    h=hashlib.sha256()
    v=model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
    f=model.mesh_face[model.mesh_faceadr[mesh]:model.mesh_faceadr[mesh]+model.mesh_facenum[mesh]]
    h.update(v.tobytes());h.update(f.tobytes());return h.hexdigest()

def geometry_inventory(model,geom_ids=None):
    shapes={};rows=[]
    for g in range(model.ngeom) if geom_ids is None else geom_ids:
        g=int(g)
        if not 0<=g<model.ngeom:raise VisibleBindingHold('stale cached geom ID outside actual model')
        b=int(model.geom_bodyid[g]);mesh=int(model.geom_dataid[g]) if model.geom_type[g]==int(mujoco.mjtGeom.mjGEOM_MESH) else None
        if mesh is not None and mesh not in shapes:shapes[mesh]=mesh_shape_sha(model,mesh)
        rows.append(dict(geom_id=g,geom_name=name(model,mujoco.mjtObj.mjOBJ_GEOM,g),body_id=b,body_name=name(model,mujoco.mjtObj.mjOBJ_BODY,b),
            ancestors=ancestry(model,b),geom_type=int(model.geom_type[g]),mesh_id=mesh,compiled_mesh_sha256=shapes.get(mesh),
            local_pos=model.geom_pos[g].tolist(),local_quat=model.geom_quat[g].tolist(),size=model.geom_size[g].tolist(),
            visible_group=int(model.geom_group[g]),collision_filters=[int(model.geom_contype[g]),int(model.geom_conaffinity[g])]))
    return dict(schema='native-visible-collision-binding-v1',model_geometry_sha256=model_geometry_sha(model),geom_count=model.ngeom,geoms=rows)

def validate_native_geometry(model,baseline,cached_target=None,checker_joint=None):
    if baseline.get('schema')!='native-visible-collision-binding-v1':raise VisibleBindingHold('missing native visible geometry baseline')
    if model_geometry_sha(model)!=baseline.get('model_geometry_sha256'):raise VisibleBindingHold('native visible/collision body/mesh/transform/filter binding changed')
    if cached_target is not None:
        geom=int(cached_target['geom_id']);actual=name(model,mujoco.mjtObj.mjOBJ_GEOM,geom) if 0<=geom<model.ngeom else None
        if actual!=cached_target['geom_name']:raise VisibleBindingHold('stale cached geom ID/name')
        joints={x['name'] for a in ancestry(model,int(model.geom_bodyid[geom])) for x in a['joints']}
        if checker_joint not in joints:raise VisibleBindingHold('checker joint is separate from target/contact geometry')
    return True

def validate_render_model(model,renderer):
    if getattr(renderer,'_model',None) is not model:raise VisibleBindingHold('recorder renderer is bound to another native model instance')

def frame_binding(model,data,frame,camera):
    return dict(frame_index=int(frame),native_model_geometry_sha256=model_geometry_sha(model),
                actual_qpos_sha256=hashlib.sha256(np.asarray(data.qpos).tobytes()).hexdigest(),
                actual_qvel_sha256=hashlib.sha256(np.asarray(data.qvel).tobytes()).hexdigest(),
                actual_sim_time=float(data.time),camera=camera)
