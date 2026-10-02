"""Native safety instrumentation must keep the real stepping method intact."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from mobiwam.reference_formal_substep import FormalSubstepMonitor, FormalSafetyStop


class FakeSim:
    def __init__(self, data):self.data=data
    def step2(self):
        self.data.qpos[0]+=0.001
        self.data.time+=0.002
        self.data.contact=[SimpleNamespace(geom1=0,geom2=1,dist=-.0002)]
        self.data.ncon=1


class FormalSubstepTest(unittest.TestCase):
    def fake_ref(self, names):
        data=SimpleNamespace(qpos=np.zeros(3),time=0.0,contact=[],ncon=0)
        model=SimpleNamespace(ngeom=len(names))
        sim=FakeSim(data)
        return SimpleNamespace(env=SimpleNamespace(sim=sim,lite_physics=True),model_data=lambda:(model,data))

    @patch('mobiwam.reference_formal_substep.mujoco.mj_id2name')
    def test_target_finger_allowed_only_during_manipulation(self, resolve):
        names=['gripper0_right_finger1_collision','target_handle']
        resolve.side_effect=lambda model,kind,i:names[i]
        ref=self.fake_ref(names);before=ref.env.sim.step2.__func__
        mon=FormalSubstepMonitor(ref,'target')
        mon.set_boundary(0,'manipulate')
        with mon:ref.env.sim.step2()
        self.assertIs(ref.env.sim.step2.__func__,before)
        self.assertEqual(len(mon.states),2)
        with tempfile.TemporaryDirectory() as directory:
            receipt=mon.save(Path(directory))
            with np.load(Path(directory)/'formal-native-substeps.npz',allow_pickle=False) as values:
                self.assertEqual(values['qpos'].shape,(2,3))
                self.assertEqual(values['step_index'].tolist(),[0])
            self.assertEqual(receipt['substeps'],1)
        mon.set_boundary(1,'navigate')
        with self.assertRaises(FormalSafetyStop):
            with mon:ref.env.sim.step2()
        self.assertIs(ref.env.sim.step2.__func__,before)
        self.assertEqual(mon.first_forbidden['phase'],'navigate')

    @patch('mobiwam.reference_formal_substep.mujoco.mj_id2name')
    def test_robot_contact_with_background_stops(self, resolve):
        names=['robot0_arm_collision','background_wall']
        resolve.side_effect=lambda model,kind,i:names[i]
        ref=self.fake_ref(names);mon=FormalSubstepMonitor(ref,'target')
        mon.set_boundary(12,'manipulate')
        with self.assertRaises(FormalSafetyStop):
            with mon:ref.env.sim.step2()
        self.assertEqual(mon.first_forbidden['contacts'][0]['geom2'],'background_wall')


if __name__=='__main__':unittest.main()
