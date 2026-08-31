import unittest

import numpy as np

from mobiwam.records import RouteType
from mobiwam.selection import PredictedCandidate, select_minimum_cost_sufficient


class RiskConstrainedSelectionTest(unittest.TestCase):
    def test_empty_admissible_set_returns_x(self):
        candidate = PredictedCandidate(
            "e0", RouteType.EXECUTE, True,
            success_probabilities=np.array([0.4, 0.5, 0.6]),
            risk_probabilities=np.array([0.0, 0.0, 0.0]),
            cost=(1, 0, 0, 0, 1, 1),
        )
        self.assertEqual(
            select_minimum_cost_sufficient([candidate], tau_success=0.8, epsilon_risk=0.1),
            RouteType.ABSTAIN,
        )

    def test_lexicographic_minimum_is_selected_after_bounds(self):
        candidates = [
            PredictedCandidate("d0", RouteType.DOCK, True, np.array([0.9, 0.9, 0.9]), np.array([0.01, 0.01, 0.01]), (2, 1, 1, 1, 1, 1)),
            PredictedCandidate("e0", RouteType.EXECUTE, True, np.array([0.9, 0.9, 0.9]), np.array([0.01, 0.01, 0.01]), (1, 0, 0, 0, 1, 1)),
        ]
        selected = select_minimum_cost_sufficient(candidates, tau_success=0.8, epsilon_risk=0.1)
        self.assertEqual(selected.candidate_id, "e0")


if __name__ == "__main__":
    unittest.main()
