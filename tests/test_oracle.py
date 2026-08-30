import unittest

from mobiwam.oracle import (
    aggregate_candidates,
    choose_best_fixed_route,
    evaluate_fixed_route,
    select_route_oracle,
    select_within_route,
)
from mobiwam.records import CandidateOutcome, RouteType


def outcome(snapshot: str, route: str, candidate: str, seed: int, success: bool):
    return CandidateOutcome(
        snapshot_id=snapshot,
        route_type=RouteType(route),
        candidate_id=candidate,
        seed=seed,
        stage_eligible=True,
        hard_valid=True,
        success=success,
        irreversible_failure=False,
        collision=False,
        contact_loss=False,
        completion_time_s=2.0,
        base_path_m=0.0 if route == "E" else 0.2,
    )


class RouteOracleTest(unittest.TestCase):
    def test_oracle_exposes_non_constant_route_preference(self):
        rows = []
        for seed in range(3):
            rows.extend(
                [
                    outcome("s1", "E", "e0", seed, True),
                    outcome("s1", "D", "d0", seed, False),
                    outcome("s2", "E", "e0", seed, False),
                    outcome("s2", "D", "d0", seed, True),
                ]
            )
        within = select_within_route(aggregate_candidates(rows))
        oracle = select_route_oracle(within)

        self.assertEqual(oracle["s1"].route_type, RouteType.EXECUTE)
        self.assertEqual(oracle["s2"].route_type, RouteType.DOCK)
        self.assertEqual(choose_best_fixed_route(within, ["s1", "s2"]), RouteType.EXECUTE)

        fixed = evaluate_fixed_route(within, RouteType.EXECUTE, ["s1", "s2"])
        oracle_success = sum(score.success_rate for score in oracle.values()) / 2
        fixed_success = sum(score.success_rate for score in fixed.values()) / 2
        self.assertEqual(oracle_success, 1.0)
        self.assertEqual(fixed_success, 0.5)

    def test_invalid_candidates_are_retained_in_raw_data_but_not_selected(self):
        valid = outcome("s1", "E", "e0", 0, True)
        invalid = CandidateOutcome(
            snapshot_id="s1",
            route_type=RouteType.ASSIST,
            candidate_id="a0",
            seed=0,
            stage_eligible=True,
            hard_valid=False,
            success=False,
            irreversible_failure=False,
            collision=False,
            contact_loss=False,
            completion_time_s=0.0,
            base_path_m=0.0,
        )
        within = select_within_route(aggregate_candidates([valid, invalid]))
        self.assertEqual([score.route_type for score in within], [RouteType.EXECUTE])


if __name__ == "__main__":
    unittest.main()
