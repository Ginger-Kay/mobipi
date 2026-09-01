import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TASKS = (
    "CloseSingleDoor",
    "CloseDrawer",
    "TurnOnFaucet",
    "TurnOnMicrowave",
)
LAYOUTS = (1, 4, 7, 8, 9)
SIGMAS = (0.0, 0.1, 0.2)
LOCKED_LAYOUTS = {
    "CloseSingleDoor": 1,
    "CloseDrawer": 8,
    "TurnOnFaucet": 4,
    "TurnOnMicrowave": 8,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class FormalC2ScheduleTest(unittest.TestCase):
    def test_freeze_and_shard_plan_bind_exact_c2_design(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = root / "reports"
            reports.mkdir()
            eligibility_tasks = {}
            for task in TASKS:
                task_root = reports / task
                task_root.mkdir()
                rows = []
                for source_index in range(60, 510):
                    layout = LAYOUTS[(source_index // len(SIGMAS)) % len(LAYOUTS)]
                    sigma = SIGMAS[source_index % len(SIGMAS)]
                    rows.append(
                        {
                            "eligible": True,
                            "environment_seed": 10000 + source_index,
                            "source_index": source_index,
                            "source_state_id": (
                                f"{task}-l{layout}-sig{sigma:.2f}"
                                f"-seed{10000 + source_index}"
                            ),
                            "snapshot_hash": f"snapshot-{task}-{source_index}",
                            "observation_hash": f"observation-{task}-{source_index}",
                            "reason": None,
                        }
                    )
                report = task_root / "eligibility.json"
                report.write_text(
                    json.dumps({"schema_version": "1.0", "sources": rows}) + "\n"
                )
                eligibility_tasks[task] = {
                    "report": str(report),
                    "report_sha256": sha256_file(report),
                    "eligible_source_count": len(rows),
                }
            eligibility_manifest = root / "eligibility-manifest.json"
            eligibility_manifest.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "pilot_indices_excluded": True,
                        "selection_uses_rollout_outcomes": False,
                        "tasks": eligibility_tasks,
                    }
                )
                + "\n"
            )
            feasibility = root / "split-feasibility.json"
            feasibility.write_text(
                json.dumps(
                    {
                        "status": "pass_task_specific_layout_holdout",
                        "selection_uses_rollout_outcomes": False,
                        "proposed_task_specific_held_out_locked_layouts": LOCKED_LAYOUTS,
                    }
                )
                + "\n"
            )
            grid = REPO / "configs" / "pilot_v2.json"
            base_configs = root / "base-configs"
            base_configs.mkdir()
            base_commit = "a" * 40
            for task in TASKS:
                (base_configs / f"{task}.json").write_text(
                    json.dumps({"code_commit": base_commit, "env_name": task}) + "\n"
                )
            (base_configs / "manifest.json").write_text(
                json.dumps(
                    {
                        "code_commit": base_commit,
                        "candidate_grid": {"sha256": sha256_file(grid)},
                    }
                )
                + "\n"
            )

            schedule_root = root / "schedule"
            code_commit = "b" * 40
            subprocess.run(
                [
                    sys.executable,
                    str(REPO / "scripts" / "freeze_formal_c2_schedule.py"),
                    "--eligibility-manifest",
                    str(eligibility_manifest),
                    "--split-feasibility-decision",
                    str(feasibility),
                    "--base-config-root",
                    str(base_configs),
                    "--candidate-grid",
                    str(grid),
                    "--output-root",
                    str(schedule_root),
                    "--code-commit",
                    code_commit,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            schedule = json.loads((schedule_root / "manifest.json").read_text())
            self.assertEqual(schedule["source_count"], 480)
            self.assertEqual(
                schedule["split_counts"],
                {"train": 264, "validation": 48, "calibration": 72, "locked_test": 96},
            )
            self.assertEqual(schedule["total_rollouts"], 12408)
            self.assertEqual(
                stable_hash(schedule["scientific_schedule"]),
                schedule["schedule_checksum"],
            )
            for task in TASKS:
                selected = schedule["tasks"][task]["selected_sources"]
                self.assertEqual(len(selected), 120)
                self.assertEqual(
                    Counter(row["split"] for row in selected),
                    Counter(
                        {"train": 66, "validation": 12, "calibration": 18, "locked_test": 24}
                    ),
                )
                locked = [row for row in selected if row["split"] == "locked_test"]
                self.assertEqual({row["layout_id"] for row in locked}, {LOCKED_LAYOUTS[task]})
                self.assertEqual(
                    Counter(row["base_noise_sigma"] for row in locked),
                    Counter({0.0: 8, 0.1: 8, 0.2: 8}),
                )
                config = json.loads((schedule_root / "configs" / f"{task}.json").read_text())
                self.assertEqual(config["code_commit"], code_commit)
                self.assertEqual(config["schedule_checksum"], schedule["schedule_checksum"])
                self.assertEqual(len(config["source_split_map"]), 120)

            shard_root = root / "shards"
            subprocess.run(
                [
                    sys.executable,
                    str(REPO / "scripts" / "plan_formal_c2_shards.py"),
                    "--schedule-manifest",
                    str(schedule_root / "manifest.json"),
                    "--candidate-grid",
                    str(grid),
                    "--output-root",
                    str(shard_root),
                    "--code-commit",
                    code_commit,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            plan = json.loads((shard_root / "manifest.json").read_text())
            self.assertEqual(plan["shard_count"], 240)
            self.assertEqual(plan["total_workers"], 12)
            self.assertEqual(plan["total_rollouts"], 12408)
            self.assertEqual(plan["phase_source_counts"]["train_validation"], 312)
            self.assertEqual(plan["phase_source_counts"]["calibration"], 72)
            self.assertEqual(plan["phase_source_counts"]["locked_deferred"], 96)
            self.assertTrue(
                all(
                    shard["launch_status"] == "sealed_until_locked_freeze"
                    for shard in plan["shards"]
                    if shard["split"] == "locked_test"
                )
            )


if __name__ == "__main__":
    unittest.main()
