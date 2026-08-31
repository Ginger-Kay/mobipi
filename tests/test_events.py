import unittest

from mobiwam.events import (
    BoundaryType,
    EventType,
    compile_option_events,
    validate_event_trace,
)
from mobiwam.records import RouteType


class OptionEventCompilerTest(unittest.TestCase):
    def test_execute_event_graph(self):
        events = compile_option_events(RouteType.EXECUTE)
        self.assertEqual(
            [event.event_type for event in events],
            [EventType.QUERY, EventType.EXECUTE, EventType.REPLAN],
        )
        self.assertEqual(events[-1].boundary, BoundaryType.COMMON_OUTCOME)
        validate_event_trace(RouteType.EXECUTE, events)

    def test_dock_has_typed_internal_and_common_boundaries(self):
        events = compile_option_events(RouteType.DOCK)
        self.assertEqual(
            [event.event_type for event in events],
            [
                EventType.MOVE,
                EventType.SETTLE,
                EventType.OBSERVE,
                EventType.RESET,
                EventType.QUERY,
                EventType.POST_DOCK_POLICY_READY,
                EventType.EXECUTE,
                EventType.REPLAN,
            ],
        )
        self.assertEqual(events[5].boundary, BoundaryType.INTERNAL)
        self.assertEqual(events[-1].boundary, BoundaryType.COMMON_OUTCOME)
        self.assertFalse(events[2].selection_time_available)
        self.assertFalse(events[4].selection_time_available)
        validate_event_trace(RouteType.DOCK, events)

    def test_assist_dispatch_is_simultaneous(self):
        events = compile_option_events(RouteType.ASSIST)
        execute, assist = events[1:3]
        self.assertEqual(execute.phase_index, assist.phase_index)
        self.assertEqual(execute.parallel_group, "base_arm_dispatch")
        self.assertEqual(assist.parallel_group, "base_arm_dispatch")
        validate_event_trace(RouteType.ASSIST, events)

    def test_abstain_compiles_to_safe_exit(self):
        events = compile_option_events(RouteType.ABSTAIN)
        self.assertEqual([event.event_type for event in events], [EventType.SAFE_EXIT])
        self.assertEqual(events[0].boundary, BoundaryType.COMMON_OUTCOME)


if __name__ == "__main__":
    unittest.main()
