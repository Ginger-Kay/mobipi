import unittest

from mobiwam.baselines import CandidateFeatures, select_geometry, select_stage_rule
from mobiwam.records import RouteType, Stage


class InterpretableBaselineTest(unittest.TestCase):
    def test_stage_rule_rejects_dock_after_contact(self):
        candidates = [
            CandidateFeatures("d0", RouteType.DOCK, True, 0.9, 0.9, 0.9, 0.0, 1.0),
            CandidateFeatures("e0", RouteType.EXECUTE, True, 0.5, 0.5, 0.5, 0.0, 0.0),
        ]
        self.assertEqual(select_stage_rule(candidates, Stage.CONTACT).candidate_id, "e0")

    def test_geometry_uses_hard_valid_candidates_only(self):
        candidates = [
            CandidateFeatures("d0", RouteType.DOCK, False, 1.0, 1.0, 1.0, 0.0, 0.1),
            CandidateFeatures("a0", RouteType.ASSIST, True, 0.7, 0.8, 0.8, 0.01, 0.2),
        ]
        self.assertEqual(select_geometry(candidates, Stage.PRECONTACT).candidate_id, "a0")


if __name__ == "__main__":
    unittest.main()
