import unittest

import numpy as np

from mobiwam.evaluation import paired_cluster_bootstrap, risk_coverage_curve


class LockedEvaluationTest(unittest.TestCase):
    def test_cluster_bootstrap_uses_source_units(self):
        effects = {f"s{index}": 0.1 + index * 0.001 for index in range(20)}
        strata = {source: ("task-a" if index < 10 else "task-b") for index, source in enumerate(effects)}
        result = paired_cluster_bootstrap(
            effects, strata=strata, resamples=10_000, seed=41
        )
        self.assertGreater(result.ci_low, 0.0)
        self.assertEqual(result.source_count, 20)
        self.assertEqual(result.resamples, 10_000)

    def test_risk_coverage_is_monotone_in_coverage(self):
        curve = risk_coverage_curve(
            success=np.array([1, 1, 0, 1], dtype=float),
            irreversible=np.array([0, 0, 1, 0], dtype=float),
            uncertainty=np.array([0.1, 0.2, 0.9, 0.3], dtype=float),
        )
        np.testing.assert_allclose(curve["coverage"], [0.25, 0.5, 0.75, 1.0])
        self.assertEqual(curve["irreversible_rate"][-1], 0.25)


if __name__ == "__main__":
    unittest.main()
