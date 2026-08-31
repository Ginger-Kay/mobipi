import tempfile
import unittest
from pathlib import Path

import numpy as np

from mobiwam.extract_features import (
    assemble_feature_arrays,
    encode_candidate_params,
    encode_source_context,
    observable_proprio_token,
    resample_induced_trajectory,
)
from mobiwam.records import (
    DataSplit,
    RouteRolloutRecord,
    RouteType,
    SourceStateRecord,
    Stage,
)


def source(split: DataSplit = DataSplit.TRAIN) -> SourceStateRecord:
    return SourceStateRecord(
        source_state_id="source-1",
        task_id="CloseSingleDoor",
        task_family="sustained_articulated_contact",
        episode_id="episode-1",
        instruction="close the door",
        stage=Stage.PRECONTACT,
        split=split,
        environment_seed=7,
        policy_name="bc_xfmr",
        policy_checkpoint_hash="checkpoint",
        simulator_version="robocasa==0.2.0",
        code_commit="commit",
        snapshot_hash="snapshot",
        observation_hash="observation",
        snapshot_path="snapshots/source-1",
    )


def rollout(
    source_record: SourceStateRecord,
    route: RouteType,
    candidate_id: str,
    state_trace_path: str,
) -> RouteRolloutRecord:
    return RouteRolloutRecord(
        schema_version="1.1",
        source_state_id=source_record.source_state_id,
        task_id=source_record.task_id,
        task_family=source_record.task_family,
        episode_id=source_record.episode_id,
        split=source_record.split,
        stage=source_record.stage,
        route_type=route,
        candidate_id=candidate_id,
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
        action_semantics_id="pandaomron-hybrid-mobile-v1",
        history_protocol_id="postdock-zero-settle-flush10-policy-reset-v1",
        transform_check_passed=True,
        restore_check_passed=True,
        stage_eligible=True,
        hard_valid=True,
        success=True,
        irreversible_failure=False,
        collision=False,
        contact_loss=False,
        task_progress_before=0.1,
        task_progress_after=0.8,
        progress_delta=0.7,
        execution_time_s=2.0,
        base_path_length_m=0.0 if route is RouteType.EXECUTE else 0.2,
        route_cost=0.0 if route is RouteType.EXECUTE else 1.0,
        source_snapshot_path=source_record.snapshot_path,
        video_path="",
        state_trace_path=state_trace_path,
        action_trace_path="action.npz",
        event_trace_path="events.json",
        labeler_version="sim-labeler-v1",
    )


class FrozenFeatureExtractionTest(unittest.TestCase):
    def test_candidate_encoding_ignores_realized_outcome_fields(self):
        base = {
            "target_offset_local_xy_m": [-0.08, 0.02],
            "dock_max_steps": 240,
            "position_tolerance_m": 0.005,
            "yaw_tolerance_rad": 0.02,
            "command_gain": 1.0,
        }
        realized = {
            **base,
            "dock_reached": True,
            "post_dock_policy_ready": True,
            "target_pose_world_xy": [99.0, 99.0],
            "history_reset_fingerprint": "label-derived",
        }
        np.testing.assert_array_equal(
            encode_candidate_params(RouteType.DOCK, base),
            encode_candidate_params(RouteType.DOCK, realized),
        )

    def test_proprio_token_excludes_generic_object_state(self):
        robot = np.arange(30, dtype=np.float64).reshape(10, 3)
        first = observable_proprio_token(
            {"robot0_eef_pos": robot, "object": np.zeros((10, 100))}
        )
        second = observable_proprio_token(
            {"robot0_eef_pos": robot, "object": np.full((10, 100), 1e9)}
        )
        np.testing.assert_array_equal(first, second)

    def test_source_context_has_three_visual_and_one_proprio_token(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "history.npz"
            images = np.ones((2, 3, 8, 8), dtype=np.float32)
            np.savez_compressed(
                path,
                robot0_agentview_left_image=images,
                robot0_agentview_right_image=images * 2,
                robot0_eye_in_hand_image=images * 3,
                robot0_eef_pos=np.ones((2, 3)),
                object=np.full((2, 5), 999.0),
            )

            def encoder(batch: np.ndarray) -> np.ndarray:
                means = batch.mean(axis=(1, 2, 3))
                return np.repeat(means[:, None], 1024, axis=1)

            context = encode_source_context(path, encoder)
            self.assertEqual(context.shape, (4, 1024))
            self.assertTrue(np.all(np.isfinite(context)))

    def test_trajectory_resampling_is_relative_and_finite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.npz"
            poses = np.repeat(np.eye(4)[None], 3, axis=0)
            poses[:, 0, 3] = [1.0, 1.5, 2.0]
            np.savez_compressed(path, eef_poses=poses)
            trajectory, valid = resample_induced_trajectory(path, horizon=5)
            self.assertEqual(valid, 1.0)
            self.assertEqual(trajectory.shape, (5, 7))
            self.assertAlmostEqual(float(trajectory[0, 0]), 0.0)
            self.assertAlmostEqual(float(trajectory[-1, 0]), 1.0)
            np.testing.assert_allclose(
                trajectory[:, 3:],
                np.repeat([[0.0, 0.0, 0.0, 1.0]], 5, axis=0),
            )

    def test_feature_rows_add_one_a0_audit_without_new_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "state.npz"
            np.savez_compressed(trace, eef_poses=np.repeat(np.eye(4)[None], 2, axis=0))
            source_record = source()
            rollouts = [
                rollout(source_record, RouteType.EXECUTE, "e0", str(trace)),
                rollout(source_record, RouteType.DOCK, "d0", str(trace)),
                rollout(source_record, RouteType.ASSIST, "a0", str(trace)),
            ]
            arrays = assemble_feature_arrays(
                [source_record],
                rollouts,
                candidate_lookup={
                    (RouteType.EXECUTE, "e0"): {"base_locked": True},
                    (RouteType.DOCK, "d0"): {
                        "target_offset_local_xy_m": [0.0, 0.0],
                        "dock_max_steps": 240,
                    },
                    (RouteType.ASSIST, "a0"): {
                        "target_offset_local_xy_m": [0.0, 0.0],
                        "fraction_toward_dock": 0.1,
                        "max_translation_m": 0.02,
                    },
                },
                contexts={source_record.source_state_id: np.ones((4, 1024))},
                allowed_splits=frozenset({DataSplit.TRAIN}),
            )
            self.assertEqual(len(arrays["source_ids"]), 4)
            self.assertEqual(int(arrays["is_a0"].sum()), 1)
            self.assertEqual(set(arrays["source_ids"]), {source_record.source_state_id})
            audit = int(np.flatnonzero(arrays["is_a0"])[0])
            self.assertEqual(arrays["route_types"][audit], "A0")
            self.assertEqual(arrays["option_ids"][audit], 2)

    def test_train_side_extraction_rejects_locked_source_before_trace_read(self):
        with self.assertRaisesRegex(ValueError, "not allowed"):
            assemble_feature_arrays(
                [source(DataSplit.LOCKED_TEST)],
                [],
                candidate_lookup={},
                contexts={"source-1": np.ones((4, 1024))},
                allowed_splits=frozenset({DataSplit.TRAIN}),
            )


if __name__ == "__main__":
    unittest.main()
