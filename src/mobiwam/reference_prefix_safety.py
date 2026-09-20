"""Sampled joint-margin guard for development D prefix prediction."""
import numpy as np


class JointMarginMonitor:
    def __init__(self, qpos_indices, limits, names):
        self.indices = np.asarray(qpos_indices, dtype=int)
        self.limits = np.asarray(limits, dtype=float)
        self.names = list(names)
        if (self.indices.ndim != 1 or not len(self.indices)
                or len(set(self.indices)) != len(self.indices) or np.any(self.indices < 0)
                or self.limits.shape != (len(self.indices), 2)
                or len(self.names) != len(self.indices)
                or not np.isfinite(self.limits).all()
                or np.any(self.limits[:, 1] - self.limits[:, 0] <= .03)):
            raise ValueError('invalid prefix joint bounds')
        self.minimum = None
        self.samples = 0

    def observe(self, qpos, *, step, substep, phase, when):
        values = np.asarray(qpos)[self.indices]
        self.samples += 1
        where = dict(step=step, substep=substep, phase=phase, when=when)
        if not np.isfinite(values).all():
            return dict(kind='predicted_nonfinite_joint_state', **where)
        margins = np.minimum(values - self.limits[:, 0], self.limits[:, 1] - values)
        joint = int(np.argmin(margins))
        margin = float(margins[joint])
        record = dict(joint=self.names[joint], margin_rad=margin, **where)
        if self.minimum is None or margin < self.minimum['margin_rad']:
            self.minimum = record
        if margin <= .015:
            return dict(kind='predicted_joint_margin', required_strictly_greater_than_rad=.015, **record)
        return None

    def receipt(self):
        return dict(required_strictly_greater_than_rad=.015, samples=self.samples,
                    minimum=self.minimum, scope='sampled states; not a continuous dynamics certificate')
