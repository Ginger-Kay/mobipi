import unittest

from mobiwam.features import ObservableCandidateInput, SimulatorOnlyLabels
from mobiwam.records import RouteType, Stage


class PrivilegedLeakageTest(unittest.TestCase):
    def test_observable_schema_rejects_simulator_labels(self):
        with self.assertRaisesRegex(ValueError, "privileged"):
            ObservableCandidateInput(
                source_state_id="s0",
                route_type=RouteType.EXECUTE,
                stage=Stage.PRECONTACT,
                policy_id="policy",
                observable_history_uri="history.npz",
                candidate_params={"object_pose": [0, 0, 0]},
                nominal_intent_uri=None,
            ).validate()

    def test_simulator_labels_are_separate(self):
        labels = SimulatorOnlyLabels(
            success=True,
            irreversible_failure=False,
            collision=False,
            contact_loss=False,
            progress=1.0,
        )
        labels.validate()


if __name__ == "__main__":
    unittest.main()
