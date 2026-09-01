import tempfile
import unittest
from pathlib import Path

from mobiwam.collection import (
    load_transactions,
    run_candidate_grid_collection,
    run_collection,
)
from mobiwam.collector import RestoreEvidence, SourceSnapshot
from mobiwam.dataset import assign_group_split, validate_paired_collection
from mobiwam.records import RouteRolloutRecord, RouteType, SourceStateRecord, Stage


class CollectionAdapter:
    def __init__(self, fail_source_index=None):
        self.fail_source_index = fail_source_index
        self.failed_once = False
        self.source_index = -1
        self.environment_seed = -1
        self.source = None

    def prepare_source_state(self, source_index, environment_seed):
        self.source_index = source_index
        self.environment_seed = environment_seed
        source_state_id = f"source-{source_index}"
        self.source = SourceStateRecord(
            source_state_id=source_state_id,
            task_id="CloseSingleDoor",
            task_family="articulated-door",
            episode_id=f"episode-{source_index}",
            instruction="close the door",
            stage=Stage.PRECONTACT,
            split=assign_group_split(source_state_id),
            environment_seed=environment_seed,
            policy_name="bc_xfmr",
            policy_checkpoint_hash="checkpoint-hash",
            simulator_version="robocasa==0.2.0",
            code_commit="commit",
            snapshot_hash=f"snapshot-{source_index}",
            observation_hash=f"observation-{source_index}",
            snapshot_path=f"snapshots/{source_state_id}.npz",
        )

    def capture_source_state(self):
        return SourceSnapshot(self.source, opaque_handle=object())

    def restore_source_state(self, snapshot):
        return RestoreEvidence(
            True, snapshot.record.snapshot_hash, snapshot.record.observation_hash
        )

    def sample_nominal_policy(self, snapshot, policy_seed):
        return (snapshot.record.source_state_id, policy_seed)

    def _rollout(self, route, policy_seed, route_seed, repeat_index):
        source = self.source
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
            repeat_index=repeat_index,
            environment_seed=source.environment_seed,
            policy_seed=policy_seed,
            route_seed=route_seed,
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
            video_path=f"videos/{source.source_state_id}/{route.value}.mp4",
            state_trace_path=f"traces/{source.source_state_id}/{route.value}-state.npz",
            action_trace_path=f"traces/{source.source_state_id}/{route.value}-action.npz",
            labeler_version="sim-labeler-v1",
        )

    def execute_e(self, snapshot, nominal_chunk, **kwargs):
        return self._rollout(RouteType.EXECUTE, **kwargs)

    def execute_d(self, snapshot, **kwargs):
        if self.source_index == self.fail_source_index and not self.failed_once:
            self.failed_once = True
            raise RuntimeError("injected D failure")
        return self._rollout(RouteType.DOCK, **kwargs)

    def execute_a(self, snapshot, nominal_chunk, **kwargs):
        return self._rollout(RouteType.ASSIST, **kwargs)


class CollectionRunnerTest(unittest.TestCase):
    def test_candidate_repeat_override_rejects_overlapping_source_seed_stride(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                ValueError, "seed_stride_per_source must be at least"
            ):
                run_candidate_grid_collection(
                    CollectionAdapter(),
                    output_root=Path(temporary),
                    source_indices=[0],
                    candidate_grid={
                        "seeds_per_candidate": 2,
                        "schedule_seed": 1,
                        "dock_candidates": [],
                        "assist_candidates": [],
                    },
                    seeds_per_candidate_override=3,
                    seed_stride_per_source=2,
                )

    def test_collection_materializes_complete_paired_manifests(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            run_collection(
                CollectionAdapter(),
                output_root=output_root,
                start_index=0,
                source_count=2,
                environment_seed_start=10,
                policy_seed_start=20,
                route_seed_start=30,
            )
            transactions = load_transactions(output_root)
            sources = [transactions[index][0] for index in sorted(transactions)]
            rollouts = [
                row for index in sorted(transactions) for row in transactions[index][1]
            ]
            report = validate_paired_collection(
                sources, rollouts, expected_source_states=2
            )
            self.assertTrue(report.ok)
            self.assertEqual(report.route_counts, {"E": 2, "D": 2, "A": 2})
            source_lines = (output_root / "manifests" / "source_states.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            rollout_lines = (output_root / "manifests" / "route_rollouts.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(source_lines), 2)
            self.assertEqual(len(rollout_lines), 6)

    def test_failed_branch_does_not_commit_partial_source_and_resume_skips_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            failing = CollectionAdapter(fail_source_index=1)
            with self.assertRaisesRegex(RuntimeError, "injected D failure"):
                run_collection(
                    failing,
                    output_root=output_root,
                    start_index=0,
                    source_count=2,
                )
            self.assertEqual(set(load_transactions(output_root)), {0})

            run_collection(
                CollectionAdapter(),
                output_root=output_root,
                start_index=0,
                source_count=2,
            )
            self.assertEqual(set(load_transactions(output_root)), {0, 1})

    def test_resume_rejects_seed_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            run_collection(
                CollectionAdapter(),
                output_root=output_root,
                start_index=0,
                source_count=1,
                environment_seed_start=10,
            )
            with self.assertRaisesRegex(RuntimeError, "resume seed mismatch"):
                run_collection(
                    CollectionAdapter(),
                    output_root=output_root,
                    start_index=0,
                    source_count=1,
                    environment_seed_start=11,
                )


if __name__ == "__main__":
    unittest.main()
