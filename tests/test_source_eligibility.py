import unittest
from types import SimpleNamespace

from mobiwam.adapters.mobipi import SourceStateIneligibleError
from mobiwam.source_eligibility import probe_source_eligibility


class FakeEligibilityAdapter:
    def __init__(self, ineligible_indices):
        self.ineligible_indices = set(ineligible_indices)
        self.current_index = None
        self.current_seed = None

    def prepare_source_state(self, source_index, environment_seed):
        self.current_index = source_index
        self.current_seed = environment_seed

    def capture_source_state(self):
        if self.current_index in self.ineligible_indices:
            raise SourceStateIneligibleError(
                f"initial mobile-base collision: environment_seed={self.current_seed}"
            )
        record = SimpleNamespace(
            source_state_id=f"source-{self.current_index}",
            snapshot_hash=f"snapshot-{self.current_index}",
            observation_hash=f"observation-{self.current_index}",
        )
        return SimpleNamespace(record=record)


class SourceEligibilityTest(unittest.TestCase):
    def test_probe_keeps_only_precontact_sources(self):
        report = probe_source_eligibility(
            FakeEligibilityAdapter({16}),
            [6, 16, 7],
            environment_seed_start=10,
        )

        self.assertEqual(report["probed_source_count"], 3)
        self.assertEqual(report["eligible_source_count"], 2)
        self.assertEqual(report["ineligible_source_count"], 1)
        self.assertEqual(report["eligible_source_indices"], [6, 7])
        self.assertFalse(report["sources"][1]["eligible"])
        self.assertIn("environment_seed=26", report["sources"][1]["reason"])

    def test_probe_rejects_duplicate_indices(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            probe_source_eligibility(FakeEligibilityAdapter(set()), [1, 1])


if __name__ == "__main__":
    unittest.main()
