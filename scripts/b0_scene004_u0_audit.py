#!/usr/bin/env python3
"""SCENE-004 U0: zero-reset diagnostic audit of the SCENE-003 records.

This program intentionally has no RoboCasa imports.  It only consumes the
append-only reset records and their already-rendered native frames.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FORBIDDEN_OUTCOME_KEYS = {
    "success",
    "task_success",
    "progress_after",
    "contact_after",
    "route_cost",
    "outcome",
    "terminal_reason",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    if not len(finite):
        return {"count": 0, "min": None, "p05": None, "median": None}
    return {
        "count": int(len(finite)),
        "min": float(np.min(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "median": float(np.median(finite)),
    }


def functional_fixture_diagnostic(row: dict[str, Any]) -> dict[str, Any]:
    fixture = row.get("fixture", {})
    task = row.get("task")
    joints = [str(name).lower() for name in fixture.get("joint_names", [])]
    joint_ranges = fixture.get("joint_ranges", {})
    qpos = fixture.get("qpos", {})
    binding_reconstructable = bool(fixture.get("name"))
    geometry_readable = bool(fixture.get("bbox_size_m")) and bool(joint_ranges) and bool(qpos)
    if task == "CloseSingleDoor":
        # SCENE-003 did not preserve MuJoCo joint types.  A task-target fixture
        # with exactly one finite-range joint is therefore the strongest
        # counterfactual reconstruction available; microwave ``microjoint``
        # names must not be rejected merely for lacking the word "hinge".
        controllable = len(joints) == 1 and len(joint_ranges) == 1
        reason = None if controllable else "not_exactly_one_finite_range_joint"
    elif task == "CloseDrawer":
        controllable = any("slide" in name or "joint" in name for name in joints)
        reason = None if controllable else "no_named_prismatic_joint_reconstructable"
    else:
        controllable, reason = False, "unsupported_task"
    passed = bool(binding_reconstructable and geometry_readable and controllable)
    return {
        "passed_counterfactual": passed,
        "reason": reason,
        "task_target_object_binding": "reconstructable_from_runtime_target_reference"
        if binding_reconstructable
        else "not_reconstructable",
        "task_success_checker_joint_binding": "not_reconstructable",
        "handle_geometry": "not_reconstructable",
        "actual_joint_type": "not_reconstructable_from_parent_record",
        "class_or_size_whitelist_used": False,
    }


def predicates(row: dict[str, Any]) -> dict[str, Any]:
    fixture = row.get("fixture", {})
    geometry = row.get("geometry", {})
    camera = row.get("camera", {})
    scene = row.get("scene_model", {})
    policy = row.get("policy_forward", {})
    functional = functional_fixture_diagnostic(row)
    d_clearance = geometry.get("d_min_clearance_m")
    a_clearance = geometry.get("a_min_clearance_m")
    intent_norm = geometry.get("nominal_intent_norm")
    joint_margin = geometry.get("joint_limit_margin")
    finite_policy = policy.get("future_chunk_sha256") is not None and policy.get("first_action_max_abs_error") is not None
    return {
        "legacy_fixture": bool(fixture.get("passed", False)),
        "functional_fixture_counterfactual": functional["passed_counterfactual"],
        "d_clearance_legacy_double_margin": d_clearance is not None and float(d_clearance) >= 0.05,
        "d_clearance_single_inflation_counterfactual": d_clearance is not None and float(d_clearance) >= 0.0,
        "a_clearance_legacy_double_margin": a_clearance is not None and float(a_clearance) >= 0.05,
        "a_clearance_single_inflation_counterfactual": a_clearance is not None and float(a_clearance) >= 0.0,
        "policy_intent": intent_norm is not None and float(intent_norm) >= 1e-4 and finite_policy,
        "joint_margin": joint_margin is not None and float(joint_margin) > 1e-4,
        "camera": bool(camera.get("passed", False)),
        "scene_binding": bool(scene.get("passed", False)),
    }


def reason_vector(preds: dict[str, Any]) -> list[str]:
    return [name for name, passed in preds.items() if not passed]


def record_id(row: dict[str, Any]) -> str:
    return f"{row.get('task')}-l{row.get('layout')}-s{int(row.get('environment_seed', -1)):02d}"


def draw_overlay(row: dict[str, Any], thumb_size: tuple[int, int] = (640, 360)) -> Image.Image:
    frame_path = Path(row["camera"]["frame_path"])
    image = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    border = (int(0.05 * width), int(0.05 * height), int(0.95 * width), int(0.95 * height))
    draw.rectangle(border, outline=(255, 64, 64), width=7)
    pixels = row.get("camera", {}).get("pixels", [])
    if len(pixels) >= 18:
        pts = [(float(p[0]), float(p[1])) for p in pixels]
        # Legacy projection order: base, D-start, dock, A-start, A-end,
        # fixture, dock corners, D-start corners, A-end corners.
        draw.line([pts[1], pts[2]], fill=(40, 220, 80), width=12)
        draw.line([pts[3], pts[4]], fill=(255, 150, 20), width=12)
        for index, colour, label in ((0, (80, 160, 255), "E/base"), (1, (40, 220, 80), "D start"),
                                      (2, (40, 220, 80), "D dock"), (4, (255, 150, 20), "A end"),
                                      (5, (255, 230, 40), "fixture")):
            x, y = pts[index]
            draw.ellipse((x - 18, y - 18, x + 18, y + 18), outline=colour, width=8)
            draw.text((x + 20, y - 12), label, fill=colour, stroke_width=3, stroke_fill=(0, 0, 0))
        # Projected corner groups visualize the inflated-footprint/context envelopes.
        for group, colour in ((pts[6:10], (40, 220, 80)), (pts[10:14], (40, 220, 80)), (pts[14:18], (255, 150, 20))):
            draw.polygon(group, outline=colour, width=6)
    preds = predicates(row)
    reasons = ", ".join(reason_vector(preds)) or "all legacy reconstructable predicates pass"
    title = f"{record_id(row)} | {row.get('fixture', {}).get('class', '').split('.')[-1]} | {row.get('status')}"
    draw.rectangle((0, 0, width, 130), fill=(0, 0, 0, 210))
    draw.text((24, 20), title, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
    draw.text((24, 68), reasons[:180], fill=(255, 220, 100), stroke_width=1, stroke_fill=(0, 0, 0))
    image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", thumb_size, "black")
    canvas.paste(image, ((thumb_size[0] - image.width) // 2, (thumb_size[1] - image.height) // 2))
    return canvas


def choose_contact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        preds = predicates(row)
        failures = reason_vector(preds)
        bucket = failures[0] if failures else "pass"
        buckets[(str(row.get("task")), bucket)].append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(buckets):
        bucket_rows = sorted(buckets[key], key=lambda row: (int(row.get("layout", 0)), int(row.get("environment_seed", 0))))
        selected.append(bucket_rows[0])
    # Keep the sheet compact and deterministic while retaining both tasks and pass/fail cases.
    return selected[:12]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    if len(rows) != 615:
        raise ValueError(f"expected exactly 615 parent records, got {len(rows)}")
    for row in rows:
        leaked = FORBIDDEN_OUTCOME_KEYS.intersection(row)
        if leaked:
            raise ValueError(f"route outcome fields are forbidden in U0: {sorted(leaked)}")
        if row.get("env_step_calls") != 0 or row.get("route_outcome_read") is not False:
            raise ValueError(f"parent record violates zero-outcome boundary: {record_id(row)}")

    detailed: list[dict[str, Any]] = []
    group_counts: dict[tuple[str, int, str], Counter[str]] = defaultdict(Counter)
    intersections: Counter[str] = Counter()
    for row in rows:
        preds = predicates(row)
        failures = reason_vector(preds)
        key = (str(row["task"]), int(row["layout"]), str(row.get("fixture", {}).get("class", "unknown")).split(".")[-1])
        group_counts[key]["records"] += 1
        group_counts[key][str(row.get("status", "unknown"))] += 1
        for name, passed in preds.items():
            group_counts[key][f"{name}_{'pass' if passed else 'fail'}"] += 1
        intersection = " & ".join(failures) if failures else "all_reconstructable_pass"
        group_counts[key][f"intersection::{intersection}"] += 1
        intersections[intersection] += 1
        functional = functional_fixture_diagnostic(row)
        detailed.append({
            "record_id": record_id(row), "task": row["task"], "cell": int(row["layout"]),
            "environment_seed": int(row["environment_seed"]), "legacy_status": row["status"],
            "fixture_name": row.get("fixture", {}).get("name"),
            "fixture_class": row.get("fixture", {}).get("class"),
            "predicates": preds, "failure_reasons": failures, "functional_fixture": functional,
            "counterfactual_eligible_on_reconstructable_fields": bool(all([
                preds["functional_fixture_counterfactual"], preds["d_clearance_single_inflation_counterfactual"],
                preds["a_clearance_single_inflation_counterfactual"], preds["policy_intent"],
                preds["joint_margin"], preds["camera"], preds["scene_binding"],
            ])),
        })

    predicate_names = sorted(predicates(rows[0]))
    csv_columns = ["task", "cell", "fixture_subtype", "records"]
    csv_columns += [f"{name}_{state}" for name in predicate_names for state in ("pass", "fail")]
    csv_rows = []
    json_groups = []
    for (task, cell, subtype), counter in sorted(group_counts.items()):
        item = {"task": task, "cell": cell, "fixture_subtype": subtype, **dict(counter)}
        csv_rows.append({column: item.get(column, 0) for column in csv_columns})
        json_groups.append(item)
    with (args.artifact_root / "legacy-failure-breakdown.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(csv_rows)

    distributions: dict[str, Any] = {}
    nearest: dict[str, Any] = {}
    for task in sorted({str(row["task"]) for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        distributions[task] = {
            "d_signed_clearance_after_footprint_plus_5cm_inflation_m": quantiles(row["geometry"]["d_min_clearance_m"] for row in task_rows),
            "a_signed_clearance_after_footprint_plus_5cm_inflation_m": quantiles(row["geometry"]["a_min_clearance_m"] for row in task_rows),
            "camera_min_border_fraction": quantiles(row["camera"]["min_border_fraction"] for row in task_rows),
            "joint_margin": quantiles(row["geometry"]["joint_limit_margin"] for row in task_rows),
        }
        nearest[task] = {
            "D": dict(Counter(str(row["geometry"]["d_nearest_obstacle"]) for row in task_rows)),
            "A": dict(Counter(str(row["geometry"]["a_nearest_obstacle"]) for row in task_rows)),
        }

    write_json(args.artifact_root / "legacy-failure-breakdown.json", {
        "record_count": len(rows), "records_sha256": sha256(args.records),
        "groups": json_groups, "failure_intersections": dict(intersections),
        "distributions": distributions, "nearest_obstacle_frequency": nearest,
        "route_outcome_reads": 0, "env_step_calls": 0, "reset_calls": 0,
    })
    write_json(args.artifact_root / "functional-fixture-reclassification.json", {
        "scope": "counterfactual diagnostic only; not a new Gate and not cell-selection evidence",
        "records": detailed,
        "summary": {
            "legacy_fixture_pass": sum(item["predicates"]["legacy_fixture"] for item in detailed),
            "functional_fixture_counterfactual_pass": sum(item["predicates"]["functional_fixture_counterfactual"] for item in detailed),
            "reconstructable_counterfactual_eligible": sum(item["counterfactual_eligible_on_reconstructable_fields"] for item in detailed),
        },
        "not_reconstructable": ["task_success_checker_joint_binding", "handle_geometry", "runtime_joint_type",
                                "actual_27_pose_candidate_search", "fixed_per_cell_camera_grid"],
    })

    cell_rows = []
    for task in sorted({str(row["task"]) for row in rows}):
        for cell in sorted({int(row["layout"]) for row in rows if row["task"] == task}):
            selected = [row for row in rows if row["task"] == task and int(row["layout"]) == cell]
            cell_rows.append({
                "task": task, "cell": cell, "record_count": len(selected),
                "legacy_eligible": sum(row["status"] == "eligible" for row in selected),
                "functional_fixture_counterfactual_pass": sum(predicates(row)["functional_fixture_counterfactual"] for row in selected),
                "single_inflation_d_pass": sum(predicates(row)["d_clearance_single_inflation_counterfactual"] for row in selected),
                "single_inflation_a_pass": sum(predicates(row)["a_clearance_single_inflation_counterfactual"] for row in selected),
                "camera_pass": sum(predicates(row)["camera"] for row in selected),
            })
    write_json(args.artifact_root / "legacy-cell-prior-diagnostic.json", {
        "use_constraint": "explanation only; MUST NOT select or eliminate any U2 cell",
        "known_implementation_defects": [
            "27-pose lattice length recorded but lattice poses did not participate in candidate search",
            "base footprint was inflated by 5 cm then signed clearance was required to be >=5 cm",
            "camera was relocated independently per reset instead of frozen per cell",
            "A direction used the first two future-action base dimensions rather than explicit EE transforms",
            "final failure status merged independent geometry and camera predicates",
        ],
        "cells": cell_rows, "route_outcome_reads": 0, "env_step_calls": 0, "reset_calls": 0,
    })

    selected = choose_contact_rows(rows)
    thumbs = [draw_overlay(row) for row in selected]
    columns = 3
    rows_n = max(1, (len(thumbs) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * 640, rows_n * 360), (18, 18, 18))
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * 640, (index // columns) * 360))
    sheet.save(args.artifact_root / "contact-sheet.png", compress_level=6)
    write_json(args.artifact_root / "u0-completion.json", {
        "status": "pass", "records": len(rows), "records_sha256": sha256(args.records),
        "selected_contact_sheet_records": [record_id(row) for row in selected],
        "contact_sheet_sha256": sha256(args.artifact_root / "contact-sheet.png"),
        "route_outcome_reads": 0, "env_step_calls": 0, "reset_calls": 0,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
