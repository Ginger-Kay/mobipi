import unittest
from collections import Counter

from mobiwam.protocol import (
    PILOT_TASKS,
    PilotConfig,
    bind_formal_source_count,
    build_pilot_schedule,
    stratified_group_split,
)
from mobiwam.records import DataSplit


class ProtocolFreezeTest(unittest.TestCase):
    def test_pilot_has_balanced_48_sources_and_full_route_support(self):
        config = PilotConfig()
        rows = build_pilot_schedule(config)
        self.assertEqual(len({row.source_state_id for row in rows}), 48)
        self.assertEqual(len(rows), 48 * (1 + 5 + 5) * 2)
        source_tasks = {
            row.source_state_id: row.task_id
            for row in rows
        }
        self.assertEqual(Counter(source_tasks.values()), Counter({task: 12 for task in PILOT_TASKS}))
        first_source = [row for row in rows if row.source_state_id == "pilot-source-000"]
        self.assertEqual(Counter(row.route_type for row in first_source), {"E": 2, "D": 10, "A": 10})
        self.assertEqual({row.order_index for row in first_source}, set(range(22)))

    def test_stratified_split_keeps_source_atomic(self):
        units = [
            (f"s{index:03d}", "task-a" if index < 20 else "task-b", str(index % 2))
            for index in range(40)
        ]
        split = stratified_group_split(units, seed=31)
        self.assertEqual(set(split), {unit[0] for unit in units})
        self.assertEqual(Counter(split.values())[DataSplit.TRAIN], 24)
        self.assertEqual(Counter(split.values())[DataSplit.VALIDATION], 6)
        self.assertEqual(Counter(split.values())[DataSplit.CALIBRATION], 4)
        self.assertEqual(Counter(split.values())[DataSplit.LOCKED_TEST], 6)

    def test_formal_n_uses_largest_feasible_bound(self):
        self.assertEqual(bind_formal_source_count(feasible_sources=580), 580)
        self.assertEqual(bind_formal_source_count(feasible_sources=900), 600)
        self.assertIsNone(bind_formal_source_count(feasible_sources=479))


if __name__ == "__main__":
    unittest.main()
