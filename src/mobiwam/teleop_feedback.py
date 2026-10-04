"""Read-only operator feedback; never emits or changes control actions."""
import numpy as np


def touching_fingers(contacts, target_geom, finger_geoms):
    """Per-finger physical contact at this sample, not a stable-grasp verdict.

    Count each finger once; ignore positive-gap proximity and other objects.
    Geometries are supplied explicitly by the loaded gripper/target model.
    """
    touching = set()
    for contact in contacts:
        distance = contact.get('distance', float('nan'))
        if not np.isfinite(distance) or distance > 0:
            continue
        a, b = contact.get('geom1'), contact.get('geom2')
        other = b if a == target_geom else a if b == target_geom else None
        if other is not None:
            touching.add(other)
    return tuple(bool(touching.intersection(geoms)) for geoms in finger_geoms)


def signed_planar_angle_deg(current, target):
    """Signed angle from current to target, around world-up, or None if tilted."""
    a,b=np.asarray(current,dtype=float)[:2],np.asarray(target,dtype=float)[:2]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    if min(np.linalg.norm(a),np.linalg.norm(b))<.25:
        return None
    return float(np.degrees(np.arctan2(a[0]*b[1]-a[1]*b[0],np.dot(a,b))))
