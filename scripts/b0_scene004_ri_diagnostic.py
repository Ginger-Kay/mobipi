#!/usr/bin/env python3
"""RI1 retained-context versus explicit-current diagnostic on frozen snapshots."""
from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-id", type=int, required=True)
    args = parser.parse_args()
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.device_id)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.device_id))
    from mobiwam.scene004_renderer import (
        context_binding,
        create_context,
        current_context_address,
        load_snapshot_sim,
        set_snapshot_state,
    )

    payload = json.loads(args.payload.read_text())
    args.output.mkdir(parents=True, exist_ok=True)

    def run_variant(name: str, explicit: bool) -> dict:
        target_sim, _ = load_snapshot_sim(Path(payload["sources"][0]["snapshot_path"]))
        target_context = create_context(target_sim, args.device_id)
        # A second low-level context creates the retained-context ordering; no
        # task environment or reset is involved.
        other_sim, _ = load_snapshot_sim(Path(payload["sources"][0]["snapshot_path"]))
        other_context = create_context(other_sim, args.device_id)
        rows = []
        try:
            for source in payload["sources"]:
                set_snapshot_state(target_sim, Path(source["snapshot_path"]))
                before = context_binding(target_context)
                current_before = current_context_address()
                if explicit:
                    target_context.gl_ctx.make_current()
                try:
                    frame = target_sim.render(camera_name="freeview", width=1920, height=1080)[::-1]
                    rows.append({
                        "source_id": source["source_id"],
                        "stratum": source["stratum"],
                        "status": "pass",
                        "shape": list(frame.shape),
                        "dtype": str(frame.dtype),
                        "before": before,
                        "current_before": current_before,
                        "after": context_binding(target_context),
                        "nonconstant": bool(frame.min() != frame.max()),
                    })
                except Exception as error:
                    rows.append({
                        "source_id": source["source_id"],
                        "stratum": source["stratum"],
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "before": before,
                        "current_before": current_before,
                        "after": context_binding(target_context),
                    })
        finally:
            target_context.gl_ctx.make_current()
            target_context.gl_ctx.free()
            other_context.gl_ctx.make_current()
            other_context.gl_ctx.free()
        return {
            "variant": name,
            "task": payload.get("task"),
            "cell": payload.get("cell"),
            "environment_seed": payload.get("environment_seed"),
            "explicit_make_current": explicit,
            "frames": rows,
            "env_reset_calls": 0,
            "env_step_calls": 0,
            "route_outcome_reads": 0,
        }

    result = run_variant("B-explicit-make-current", True)
    (args.output / "B-explicit-make-current.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    retained = run_variant("A-retained-current-path", False)
    (args.output / "A-retained-current-path.json").write_text(json.dumps(retained, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
