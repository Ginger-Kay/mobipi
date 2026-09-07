"""V3.1 bounded geometry/runtime development entry point (no outcome search)."""
import argparse
import os
from pathlib import Path

import mujoco
import numpy as np

from v2_policy_a_repair_video import make_adapter, code_commit
from mobiwam.v3_control import LiveControl
from mobiwam.v3_pose_compiler import write


def audit(root, task):
    adapter, snapshot, source = make_adapter(root / 'adapter', task, 'qualification', False)
    receipt = adapter.restore_source_state(snapshot)
    if not receipt.passed:
        raise RuntimeError('restore failed')
    c = LiveControl(adapter)
    m, d = c.model, c.data
    pairs = []
    for row in c.clearance()[:20]:
        ids = [c.names.index(n) for n in row['pair']]
        geoms = []
        for i in ids:
            geoms.append(dict(id=i, name=c.names[i], body_id=int(m.geom_bodyid[i]),
                type=int(m.geom_type[i]), size=m.geom_size[i], world_position=d.geom_xpos[i],
                world_rotation=d.geom_xmat[i], contype=int(m.geom_contype[i]),
                conaffinity=int(m.geom_conaffinity[i]), margin=float(m.geom_margin[i]), gap=float(m.geom_gap[i])))
        explicit = [i for i in range(m.npair) if set((int(m.pair_geom1[i]), int(m.pair_geom2[i]))) == set(ids)]
        contacts = [dict(distance=float(d.contact[i].dist), exclude=int(d.contact[i].exclude))
                    for i in range(d.ncon) if set(map(int, d.contact[i].geom)) == set(ids)]
        pairs.append(dict(**row, geoms=geoms, explicit_pairs=explicit, live_contacts=contacts,
            distance_source='mj_geomDistance signed surface distance, no model margin added',
            project_margin_applications=1))
    write(root / task / 'pair-audit.json', dict(code_commit=code_commit(), source=source,
        pid=os.getpid(), env_steps=0, pairs=pairs, exclusions=m.exclude_signature,
        context=c.context, base_qpos_indices=c.base.qpos_index, arm_qpos_indices=c.arm.qpos_index,
        base_qpos=d.qpos[c.base.qpos_index], arm_qpos=d.qpos[c.arm.qpos_index],
        eef_position=d.site_xpos[c.site], eef_rotation=d.site_xmat[c.site], handle_pose=c.handle_pose(),
        note='2mm arm margin and 1cm IK tolerance inherited implementation criteria, not scientific definitions'))
    print(task, 'pair audit saved', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['audit'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--task', choices=['CloseDrawer', 'CloseSingleDoor'], required=True)
    args = p.parse_args()
    audit(args.root, args.task)
