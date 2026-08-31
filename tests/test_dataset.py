import unittest

from mobiwam.dataset import (
    assign_group_split,
    derive_abstain_targets,
    validate_paired_collection,
)
from mobiwam.records import (
    DataSplit,
    RouteRolloutRecord,
    RouteType,
    SourceStateRecord,
    Stage,
)


def source(source_state_id: str) -> SourceStateRecord:
    return SourceStateRecord(
        source_state_id=source_state_id,
        task_id="CloseSingleDoor",
        task_family="articulated-door",
        episode_id=f"episode-{source_state_id}",
        instruction="close the door",
        stage=Stage.PRECONTACT,
        split=assign_group_split(source_state_id),
        environment_seed=7,
        policy_name="bc_xfmr",
        policy_checkpoint_hash="checkpoint-sha256",
        simulator_version="robocasa==0.2.0",
        code_commit="19b130b8",
        snapshot_hash=f"snapshot-{source_state_id}",
        observation_hash=f"observation-{source_state_id}",
        snapshot_path=f"snapshots/{source_state_id}.npz",
    )


def rollout(
    source_record: SourceStateRecord,
    route: RouteType,
    *,
    success: bool = True,
) -> RouteRolloutRecord:
    return RouteRolloutRecord(
        schema_version="1.0",
        source_state_id=source_record.source_state_id,
        task_id=source_record.task_id,
        task_family=source_record.task_family,
        episode_id=source_record.episode_id,
        split=source_record.split,
        stage=source_record.stage,
        route_type=route,
        candidate_id=f"{route.value.lower()}0",
        repeat_index=0,
        environment_seed=source_record.environment_seed,
        policy_seed=11,
        route_seed=13,
        policy_name=source_record.policy_name,
        policy_checkpoint_hash=source_record.policy_checkpoint_hash,
        simulator_version=source_record.simulator_version,
        code_commit=source_record.code_commit,
        snapshot_hash=source_record.snapshot_hash,
        observation_hash=source_record.observation_hash,
        action_semantics_id="mobipi-12d-v1",
        history_protocol_id="settled-repeat-v1",
        transform_check_passed=True,
        restore_check_passed=True,
        stage_eligible=True,
        hard_valid=True,
        success=success,
        irreversible_failure=False,
        collision=False,
        contact_loss=False,
        task_progress_before=0.1,
        task_progress_after=0.8 if success else 0.2,
        progress_delta=0.7 if success else 0.1,
        execution_time_s=2.0,
        base_path_length_m=0.0 if route is RouteType.EXECUTE else 0.2,
        route_cost=0.0 if route is RouteType.EXECUTE else 1.0,
        source_snapshot_path=source_record.snapshot_path,
        video_path=f"videos/{source_record.source_state_id}/{route.value}.mp4",
        state_trace_path=f"traces/{source_record.source_state_id}/{route.value}-state.npz",
        action_trace_path=f"traces/{source_record.source_state_id}/{route.value}-action.npz",
        labeler_version="sim-labeler-v1",
    )


class PairedDatasetTest(unittest.TestCase):
    def test_complete_paired_collection_passes(self):
        sources = [source("s1"), source("s2")]
        rows = [
            rollout(source_record, route)
            for source_record in sources
            for route in (RouteType.EXECUTE, RouteType.DOCK, RouteType.ASSIST)
        ]
        report = validate_paired_collection(
            sources,
            rows,
            expected_source_states=2,
        )
        self.assertTrue(report.ok)
        self.assertEqual(report.route_counts, {"E": 2, "D": 2, "A": 2})
        self.assertEqual(report.rollout_records, 6)

    def test_execute_hold_uses_max_displacement_not_cumulative_path(self):
        source_record = source("s1")
        rows = [
            rollout(source_record, route)
            for route in (RouteType.EXECUTE, RouteType.DOCK, RouteType.ASSIST)
        ]
        execute = rows[0]
        rows[0] = RouteRolloutRecord(
            **{
                **execute.__dict__,
                "base_path_length_m": 0.002,
                "candidate_params": {"max_base_displacement_m": 0.0004},
            }
        )
        report = validate_paired_collection(
            [source_record],
            rows,
            expected_source_states=1,
        )
        self.assertTrue(report.ok)

    def test_execute_max_displacement_at_one_mm_is_rejected(self):
        source_record = source("s1")
        rows = [
            rollout(source_record, route)
            for route in (RouteType.EXECUTE, RouteType.DOCK, RouteType.ASSIST)
        ]
        execute = rows[0]
        rows[0] = RouteRolloutRecord(
            **{
                **execute.__dict__,
                "candidate_params": {"max_base_displacement_m": 0.001},
            }
        )
        report = validate_paired_collection(
            [source_record],
            rows,
            expected_source_states=1,
        )
        self.assertFalse(report.ok)
        self.assertIn("execute_base_moved", {issue.code for issue in report.issues})

    def test_missing_route_is_rejected(self):
        source_record = source("s1")
        rows = [
            rollout(source_record, RouteType.EXECUTE),
            rollout(source_record, RouteType.DOCK),
        ]
        report = validate_paired_collection(
            [source_record],
            rows,
            expected_source_states=1,
        )
        self.assertFalse(report.ok)
        self.assertIn("missing_paired_branch", {issue.code for issue in report.issues})

    def test_candidate_support_mismatch_is_rejected(self):
        source_record = source("s1")
        rows = [
            rollout(source_record, route)
            for route in (RouteType.EXECUTE, RouteType.DOCK, RouteType.ASSIST)
        ]
        report = validate_paired_collection(
            [source_record],
            rows,
            expected_source_states=1,
            expected_candidates_by_route={RouteType.DOCK: ["d0", "d1"]},
        )
        self.assertFalse(report.ok)
        self.assertIn("candidate_support_mismatch", {issue.code for issue in report.issues})

    def test_branch_split_mismatch_is_rejected(self):
        source_record = source("s1")
        rows = [
            rollout(source_record, RouteType.EXECUTE),
            rollout(source_record, RouteType.DOCK),
            rollout(source_record, RouteType.ASSIST),
        ]
        wrong = rows[-1]
        alternative = (
            DataSplit.LOCKED_TEST
            if source_record.split is not DataSplit.LOCKED_TEST
            else DataSplit.TRAIN
        )
        rows[-1] = RouteRolloutRecord(
            **{**wrong.__dict__, "split": alternative}
        )
        report = validate_paired_collection(
            [source_record],
            rows,
            expected_source_states=1,
        )
        self.assertFalse(report.ok)
        self.assertIn("branch_metadata_mismatch", {issue.code for issue in report.issues})

    def test_frozen_stratified_split_map_overrides_legacy_hash_assignment(self):
        original = source("s1")
        alternative = (
            DataSplit.LOCKED_TEST
            if original.split is not DataSplit.LOCKED_TEST
            else DataSplit.TRAIN
        )
        source_record = SourceStateRecord(**{**original.__dict__, "split": alternative})
        rows = [
            rollout(source_record, route)
            for route in (RouteType.EXECUTE, RouteType.DOCK, RouteType.ASSIST)
        ]
        legacy_report = validate_paired_collection(
            [source_record], rows, expected_source_states=1
        )
        self.assertIn(
            "noncanonical_group_split",
            {issue.code for issue in legacy_report.issues},
        )

        formal_report = validate_paired_collection(
            [source_record],
            rows,
            expected_source_states=1,
            expected_source_splits={source_record.source_state_id: alternative.value},
        )
        self.assertTrue(formal_report.ok)

    def test_frozen_split_map_must_exactly_cover_collection_sources(self):
        source_record = source("s1")
        rows = [
            rollout(source_record, route)
            for route in (RouteType.EXECUTE, RouteType.DOCK, RouteType.ASSIST)
        ]
        report = validate_paired_collection(
            [source_record],
            rows,
            expected_source_states=1,
            expected_source_splits={"another-source": DataSplit.TRAIN},
        )
        codes = {issue.code for issue in report.issues}
        self.assertIn("missing_expected_source_split", codes)
        self.assertIn("unexpected_source_split_map_entries", codes)

    def test_x_is_derived_instead_of_executed(self):
        source_record = source("s1")
        failed = [
            rollout(source_record, route, success=False)
            for route in (RouteType.EXECUTE, RouteType.DOCK, RouteType.ASSIST)
        ]
        self.assertEqual(derive_abstain_targets(failed), {"s1": True})
        with self.assertRaisesRegex(ValueError, "derived decision label"):
            RouteRolloutRecord(
                **{**failed[0].__dict__, "route_type": RouteType.ABSTAIN}
            ).validate()


if __name__ == "__main__":
    unittest.main()
