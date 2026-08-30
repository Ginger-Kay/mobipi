import unittest
from dataclasses import replace

from mobiwam.collector import (
    PairedBranchCollector,
    RestoreEvidence,
    SourceSnapshot,
)
from mobiwam.dataset import assign_group_split
from mobiwam.records import RouteRolloutRecord, RouteType, SourceStateRecord, Stage


def make_source() -> SourceStateRecord:
    source_state_id = "paired-s1"
    return SourceStateRecord(
        source_state_id=source_state_id,
        task_id="CloseSingleDoor",
        task_family="articulated-door",
        episode_id="episode-1",
        instruction="close the door",
        stage=Stage.PRECONTACT,
        split=assign_group_split(source_state_id),
        environment_seed=3,
        policy_name="bc_xfmr",
        policy_checkpoint_hash="checkpoint-hash",
        simulator_version="robocasa==0.2.0",
        code_commit="commit",
        snapshot_hash="state-hash",
        observation_hash="obs-hash",
        snapshot_path="snapshot.npz",
    )


def make_rollout(source: SourceStateRecord, route: RouteType) -> RouteRolloutRecord:
    return RouteRolloutRecord(
        schema_version="1.0",
        source_state_id=source.source_state_id,
        task_id=source.task_id,
        task_family=source.task_family,
        episode_id=source.episode_id,
        split=source.split,
        stage=source.stage,
        route_type=route,
        candidate_id=f"{route.value.lower()}0",
        repeat_index=0,
        environment_seed=source.environment_seed,
        policy_seed=5,
        route_seed=7,
        policy_name=source.policy_name,
        policy_checkpoint_hash=source.policy_checkpoint_hash,
        simulator_version=source.simulator_version,
        code_commit=source.code_commit,
        snapshot_hash=source.snapshot_hash,
        observation_hash=source.observation_hash,
        action_semantics_id="mobipi-12d-v1",
        history_protocol_id="settled-repeat-v1",
        transform_check_passed=True,
        restore_check_passed=True,
        stage_eligible=True,
        hard_valid=True,
        success=True,
        irreversible_failure=False,
        collision=False,
        contact_loss=False,
        task_progress_before=0.0,
        task_progress_after=1.0,
        progress_delta=1.0,
        execution_time_s=1.0,
        base_path_length_m=0.0 if route is RouteType.EXECUTE else 0.1,
        route_cost=0.0 if route is RouteType.EXECUTE else 1.0,
        source_snapshot_path=source.snapshot_path,
        video_path=f"videos/{route.value}.mp4",
        state_trace_path=f"traces/{route.value}-state.npz",
        action_trace_path=f"traces/{route.value}-action.npz",
        labeler_version="sim-labeler-v1",
    )


class FakeAdapter:
    def __init__(self):
        self.source = make_source()
        self.calls = []
        self.nominal = object()

    def prepare_source_state(self, source_index, environment_seed):
        self.calls.append(("prepare", source_index, environment_seed))

    def capture_source_state(self):
        self.calls.append("capture")
        return SourceSnapshot(self.source, opaque_handle="opaque")

    def restore_source_state(self, snapshot):
        self.calls.append("restore")
        return RestoreEvidence(True, self.source.snapshot_hash, self.source.observation_hash)

    def sample_nominal_policy(self, snapshot, policy_seed):
        self.calls.append(("sample", policy_seed))
        return self.nominal

    def execute_e(self, snapshot, nominal_chunk, **kwargs):
        self.calls.append(("E", nominal_chunk))
        return replace(
            make_rollout(self.source, RouteType.EXECUTE),
            policy_seed=kwargs["policy_seed"],
            route_seed=kwargs["route_seed"],
            repeat_index=kwargs["repeat_index"],
        )

    def execute_d(self, snapshot, **kwargs):
        self.calls.append("D_requery_after_dock")
        return replace(
            make_rollout(self.source, RouteType.DOCK),
            policy_seed=kwargs["policy_seed"],
            route_seed=kwargs["route_seed"],
            repeat_index=kwargs["repeat_index"],
        )

    def execute_a(self, snapshot, nominal_chunk, **kwargs):
        self.calls.append(("A", nominal_chunk))
        return replace(
            make_rollout(self.source, RouteType.ASSIST),
            policy_seed=kwargs["policy_seed"],
            route_seed=kwargs["route_seed"],
            repeat_index=kwargs["repeat_index"],
        )


class CollectorTest(unittest.TestCase):
    def test_e_and_a_share_one_nominal_chunk_and_every_route_restores(self):
        adapter = FakeAdapter()
        source, rows = PairedBranchCollector(adapter).collect_one(
            policy_seed=5,
            route_seed=7,
        )
        self.assertEqual(source.source_state_id, "paired-s1")
        self.assertEqual([row.route_type.value for row in rows], ["E", "D", "A"])
        self.assertEqual(adapter.calls.count("restore"), 3)
        self.assertEqual(adapter.calls.count(("sample", 5)), 1)
        self.assertIs(adapter.calls[3][1], adapter.calls[-1][1])
        self.assertEqual(adapter.calls[4], "restore")
        self.assertEqual(adapter.calls[5], "D_requery_after_dock")

    def test_failed_restore_stops_collection(self):
        adapter = FakeAdapter()

        def failed_restore(snapshot):
            return RestoreEvidence(False, "bad", "bad")

        adapter.restore_source_state = failed_restore
        with self.assertRaisesRegex(RuntimeError, "before route E"):
            PairedBranchCollector(adapter).collect_one(policy_seed=5, route_seed=7)

    def test_execute_base_motion_stops_before_dock(self):
        adapter = FakeAdapter()

        def moved_execute(snapshot, nominal_chunk, **kwargs):
            del snapshot, nominal_chunk
            return replace(
                make_rollout(adapter.source, RouteType.EXECUTE),
                candidate_params={"max_base_displacement_m": 0.001},
                policy_seed=kwargs["policy_seed"],
                route_seed=kwargs["route_seed"],
                repeat_index=kwargs["repeat_index"],
            )

        adapter.execute_e = moved_execute
        with self.assertRaisesRegex(RuntimeError, "strictly below 1 mm"):
            PairedBranchCollector(adapter).collect_one(policy_seed=5, route_seed=7)
        self.assertNotIn("D_requery_after_dock", adapter.calls)

    def test_multiple_repeats_reuse_one_captured_snapshot(self):
        adapter = FakeAdapter()
        collector = PairedBranchCollector(adapter)
        snapshot = adapter.capture_source_state()
        first = collector.collect_from_snapshot(
            snapshot, policy_seed=5, route_seed=7, repeat_index=0
        )
        second = collector.collect_from_snapshot(
            snapshot, policy_seed=6, route_seed=8, repeat_index=1
        )
        self.assertEqual([row.route_type.value for row in first], ["E", "D", "A"])
        self.assertEqual([row.route_type.value for row in second], ["E", "D", "A"])
        self.assertEqual({row.repeat_index for row in first}, {0})
        self.assertEqual({row.repeat_index for row in second}, {1})
        self.assertEqual(adapter.calls.count("capture"), 1)


if __name__ == "__main__":
    unittest.main()
