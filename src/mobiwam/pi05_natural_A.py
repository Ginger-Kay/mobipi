"""SIM-v2 A: let the existing constrained QP share policy EEF tracking.

No requested lateral destination, new contact constraint, gap gate, minimum
distance or minimum duration. Original A3 preload and physical guards remain.
"""
import numpy as np
from mobiwam.pi05_A3 import A3Driver


class NaturalADriver(A3Driver):
    def base_control(self):
        _,data=self.ref.model_data();base=self.ref.robot.part_controllers['base']
        if not self.a_started:
            self.events.append(dict(event='A_base_available_to_existing_QP_policy_EEF_tracking',
                                    sim_time=float(data.time),base_goal_source='current measured pose; no side displacement request'))
            self.a_started=True
        # The existing QP retains its zero-base regularizer. Base is free to
        # help track actual policy intent; no preselected side goal pulls it.
        return data.qpos[base.qpos_index].copy(),False

    def receipt(self, queries):
        old=super().receipt(queries)
        old['historical_strict_A_semantics_observed']=old['A_semantics_observed']
        old['A_semantics_observed']=None
        old['A_private_version']='A3N-natural-v1'
        old['A_definition']='Natural motion from original whole-body QP tracking fixed policy EEF intent; no numeric movement/contact qualification gate'
        old['base_goal_source']='current measured base pose; baseline candidate still checked for geometric availability, unused as a displacement target'
        old['native_contact_is_observation_only']=True
        old['physical_protections_unchanged']=True
        return old
