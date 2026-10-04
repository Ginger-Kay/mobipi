"""Read-only operator feedback; never emits or changes control actions."""
import numpy as np


def signed_planar_angle_deg(current, target):
    """Signed angle from current to target, around world-up, or None if tilted."""
    a,b=np.asarray(current,dtype=float)[:2],np.asarray(target,dtype=float)[:2]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    if min(np.linalg.norm(a),np.linalg.norm(b))<.25:
        return None
    return float(np.degrees(np.arctan2(a[0]*b[1]-a[1]*b[0],np.dot(a,b))))
