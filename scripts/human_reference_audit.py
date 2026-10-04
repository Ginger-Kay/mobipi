"""Development human-A reference checks; never create a primary route or train."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
import h5py
import numpy as np


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_human_a(result, metadata, actions, states):
    if result.get("route") != "A" or metadata.get("route") != "A":
        raise ValueError("Only human A replay is supported")
    if metadata.get("record_type") != "practice" or metadata.get("version") != "human-scene-pilot-v1":
        raise ValueError("Expected original development human practice")
    n = result.get("steps")
    if type(n) is not int or n < 1 or actions.shape != (n, 12) or states.shape[0] != n + 1:
        raise ValueError("Incomplete action/state alignment")
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError("Nonfinite recorded state/action")
    events = result.get("events", [])
    if not events or events[0].get("event") != "collection_begin":
        raise ValueError("Missing human collection event")
    last = -1
    paused = False
    for i, event in enumerate(events):
        kind, step = event.get("event"), event.get("step")
        if type(step) is not int or not 0 <= step <= n or step < last:
            raise ValueError("Invalid human event boundary")
        if kind == "collection_begin":
            if i != 0 or step != 0 or event.get("operator_id") != metadata.get("operator_id"):
                raise ValueError("Mismatched human collection provenance")
        elif kind == "pause":
            if paused:
                raise ValueError("Repeated pause")
            paused = True
        elif kind == "resume":
            if not paused:
                raise ValueError("Resume without pause")
            paused = False
        elif kind != "contact":
            raise ValueError("Unknown human event; cannot silently discard controller transitions")
        last = step
    # A pause changes wall-clock scheduling only. All physical inputs, including
    # gripper and base mode, are in saved actions; there are no D reset events.
    return events


def validate_human_recording(result, metadata, actions, states):
    if metadata.get("record_type") == "primary":
        from mobiwam.human_primary import verify_freeze
        receipt = json.loads(Path(metadata["freeze_receipt"]).read_text())
        config = receipt["config"]
        verify_freeze(config)
        if (metadata.get("config_version") != config["config_version"]
                or metadata.get("source_id") != Path(config["source"]).name
                or result.get("source") != config["source"]
                or metadata.get("paired_protocol_version") != config["paired_protocol_version"]
                or not result.get("events")
                or result["events"][0].get("record_type") != "primary"):
            raise ValueError("Primary recording/freeze provenance differs")
        # The following strict scheduling validator is shared with practice.
        # This local view changes no stored metadata, events or actions.
        metadata = dict(metadata, record_type="practice")
    if result.get("route") == "E":
        if metadata.get("route") != "E" or metadata.get("paired_protocol_version") != "human-eda-v2-stowed":
            raise ValueError("Unsupported E protocol")
        validate_human_a(dict(result,route="A"),dict(metadata,route="A"),actions,states)
        if np.any(actions[:,7:10] != 0) or np.any(actions[:,11] != -1):
            raise ValueError("E base input is not locked")
        return result["events"]
    if result.get("route") == "A":
        return validate_human_a(result, metadata, actions, states)
    if result.get("route") != "D" or metadata.get("route") != "D" or metadata.get("paired_protocol_version") != "human-eda-v2-stowed":
        raise ValueError("Unsupported human route/protocol")
    events = result.get("events", [])
    docks = [e for e in events if e.get("event") == "docked"]
    observations = [e for e in events if e.get("event") == "docked_state"]
    if not docks and not observations:
        if metadata.get("human_selected_dock") is not None:
            raise ValueError("Dock metadata without event")
        validate_human_a(dict(result,route="A"),dict(metadata,route="A"),actions,states)
        if any(e.get("event") == "contact" for e in events) or np.any(actions[:,:6] != 0) or np.any(actions[:,6] != -1):
            raise ValueError("D manipulation input before dock")
        return events
    if len(docks) != 1 or len(observations) != 1:
        raise ValueError("D requires one dock and one dock observation")
    dock, observation = docks[0], observations[0]
    step = dock.get("step")
    if type(step) is not int or not 0 <= step < result["steps"] or observation.get("step") != step:
        raise ValueError("Dock boundary mismatch")
    if events.index(observation) != events.index(dock)+1:
        raise ValueError("Dock observation must immediately follow marker")
    boundaries = [e.get("step") for e in events]
    if any(type(x) is not int for x in boundaries) or boundaries != sorted(boundaries):
        raise ValueError("Unordered event boundary")
    if any(e.get("event") == "contact" and e["step"] < step for e in events):
        raise ValueError("Contact marker before D dock")
    # Reuse strict human scheduling validation after explicitly interpreting D
    # markers. F5 changes the input filter only; it does not reset controllers.
    basic = dict(result, route="A", events=[e for e in events if e.get("event") not in ("docked","docked_state")])
    validate_human_a(basic, dict(metadata, route="A"), actions, states)
    velocity = np.asarray(observation.get("base_qvel"), dtype=float)
    position = np.asarray(observation.get("base_qpos"), dtype=float)
    if velocity.shape != (3,) or position.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("Invalid dock state")
    if np.any(np.abs(velocity) > [.01,.01,.02]):
        raise ValueError("Dock was not stopped")
    saved = metadata.get("human_selected_dock", {})
    if saved.get("step") != step or saved.get("base_qpos") != observation["base_qpos"] or saved.get("base_qvel") != observation["base_qvel"]:
        raise ValueError("Dock metadata differs")
    if np.any(actions[:step,:6] != 0) or np.any(actions[:step,6] != -1):
        raise ValueError("D manipulation input before dock")
    if np.any(actions[step:,7:10] != 0) or np.any(actions[step:,11] != -1):
        raise ValueError("D base input after dock")
    return events


def phase_at(result, step):
    if result["route"] == "D":
        dock = next((e["step"] for e in result["events"] if e["event"] == "docked"), result["steps"])
        if step < dock:
            return "navigate"
    return "manipulate"


def load_attempt(attempt):
    result = json.loads((attempt / "result.json").read_text())
    metadata = json.loads((attempt / "collection-metadata.json").read_text())
    with h5py.File(attempt / "demo.hdf5") as f:
        g = f["data/demo_0"]
        actions, states, integration = g["actions"][:], g["states"][:], g["initial_integration"][:]
    events = validate_human_recording(result, metadata, actions, states)
    return result, metadata, actions, states, integration, events


def load_partial_tail(attempt, result):
    partial = attempt / "partial-control-step.npz"
    safety = attempt / "safety-stop.json"
    if not partial.exists() and not safety.exists():
        if result.get("reason") in ("native_forbidden_contact_stop", "joint_margin_stop"):
            raise ValueError("Safety stop missing partial-step evidence")
        return None
    if not partial.exists() or not safety.exists():
        raise ValueError("Incomplete safety tail")
    record = json.loads(safety.read_text())
    with np.load(partial) as z:
        tail = {k: z[k] for k in ("initial_integration","terminal_integration","attempted_action")}
    if record["step"] != result["steps"] or record["failure"]["step"] != result["steps"]:
        raise ValueError("Partial-step boundary differs")
    if tail["attempted_action"].shape != (12,) or any(not np.isfinite(v).all() for v in tail.values()):
        raise ValueError("Invalid partial-step numeric evidence")
    if tail["initial_integration"].shape != tail["terminal_integration"].shape:
        raise ValueError("Partial integration shape mismatch")
    tail["record"] = record
    return tail


def replay(attempt, output, pilot_path, video=False):
    import mujoco
    import imageio.v2 as imageio
    from human_scene_pilot import PilotReference, make_args
    from mobiwam.reference_formal_substep import FormalSubstepMonitor, FormalSafetyStop
    from mobiwam.reference_prefix_safety import JointMarginMonitor, GuardedIntegration, JointMarginStop
    from mobiwam.replay_diagnostics import state_fields, summarize_drift
    result, metadata, actions, states, integration, events = load_attempt(attempt)
    pilot = json.loads(pilot_path.read_text())
    source = attempt.parent.parent
    if Path(pilot["source"]).resolve() != source or pilot["config_version"] != metadata["config_version"]:
        raise ValueError("Pilot/recording Source version differs")
    ref = PilotReference(make_args(output / "environment", pilot["task"],
                                   pilot["environment_seed"], source, interactive=False), pilot)
    actual, errors, success = [], [], []
    guard = joint = None
    stop = None
    tail = load_partial_tail(attempt, result)
    tail_check = None
    started = stamp()
    try:
        ref.route = result["route"]
        ref.evidence_camera = pilot["main_camera"]
        ref.restore()
        ref.bind()
        initial = ref.env.sim.get_state().flatten().copy()
        restore_error = float(np.max(np.abs(initial - states[0])))
        integration_error = float(np.max(np.abs(ref.integration() - integration)))
        write(output / "restore.json", dict(state_error=restore_error, integration_error=integration_error,
              source=str(source), receipt=ref.restore_receipt))
        if max(restore_error, integration_error) > 1e-10:
            raise ValueError("Original initial state/integration differs")
        m, d = ref.model_data()
        guard = FormalSubstepMonitor(ref, ref.native["fixture_name"])
        guard.set_boundary(0, "precontact")
        if list(guard.forbidden_contacts()):
            raise ValueError("Original initial native contact unsafe")
        arm = ref.robot.part_controllers["right"]
        jids = [int(np.flatnonzero(m.jnt_qposadr == i)[0]) for i in arm.qpos_index]
        joint = JointMarginMonitor(arm.qpos_index, m.jnt_range[jids],
                    [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in jids])
        actual.append(initial)
        ref.env._get_observations(force_update=True)
        write(output / "human-events.json", dict(events=events, interpretation="Original human markers retained. D F5 switches navigation to manipulation; recorded actions already contain the filter. Pause/resume have zero simulated duration; no controller reset or correction."))
        with (imageio.get_writer(output / "replay.mp4", fps=20, codec="libx264", quality=7,
                                macro_block_size=None) if video else nullcontext(None)) as writer:
            for i, action in enumerate(actions):
                phase = phase_at(result, i)
                guard.set_boundary(i, phase)
                try:
                    with guard:
                        with GuardedIntegration(ref.env.sim, d, joint, lite_physics=ref.env.lite_physics,
                                                step=i, phase=phase):
                            ref.env.step(action)
                except (FormalSafetyStop, JointMarginStop) as exc:
                    stop = dict(step=i, failure=exc.failure)
                    np.savez_compressed(output / "partial-control-step.npz",
                                        terminal_integration=ref.integration(), attempted_action=action)
                    break
                value = ref.env.sim.get_state().flatten().copy()
                actual.append(value)
                errors.append(float(np.max(np.abs(value - states[i+1]))))
                success.append(bool(ref.env._check_success()))
                ref.env._get_observations(force_update=True)
                if writer is not None:
                    writer.append_data(ref.frame(pilot["main_camera"]))
                if i % 100 == 0:
                    status = dict(at=stamp(), complete_steps=i+1, total=len(actions),
                                  state_error=errors[-1], checker_success=success[-1])
                    write(output / "progress.json", status)
                    print("HUMAN_REPLAY", result["route"], json.dumps(status), flush=True)
        if tail is not None and stop is None and len(errors) == len(actions):
            # Replay the originally attempted residual action once, from the
            # organically reached state. Never inject its terminal state.
            i = len(actions)
            initial_error = float(np.max(np.abs(ref.integration()-tail["initial_integration"])))
            if initial_error > 1e-10:
                raise ValueError("Partial-step start integration mismatch")
            phase = phase_at(result, i)
            guard.set_boundary(i, phase)
            try:
                with guard:
                    with GuardedIntegration(ref.env.sim, d, joint, lite_physics=ref.env.lite_physics,
                                            step=i, phase=phase):
                        ref.env.step(tail["attempted_action"])
            except (FormalSafetyStop, JointMarginStop) as exc:
                stop = dict(step=i, failure=exc.failure)
            terminal_error = float(np.max(np.abs(ref.integration()-tail["terminal_integration"])))
            same_failure = stop is not None and stop["failure"] == tail["record"]["failure"]
            tail_check = dict(initial_integration_error=initial_error,terminal_integration_error=terminal_error,
                expected_stop=tail["record"],actual_stop=stop,same_failure=same_failure,
                reproduced=bool(same_failure and terminal_error <= 1e-10))
            write(output / "partial-tail-replay.json",tail_check)
            np.savez_compressed(output / "partial-control-step.npz",terminal_integration=ref.integration(),
                                attempted_action=tail["attempted_action"])
        actual = np.asarray(actual)
        expected = states[:len(actual)]
        np.savez_compressed(output / "replayed-states-and-field-errors.npz",
                            actual=actual, absolute_field_errors=np.abs(actual-expected))
        write(output / "field-errors.json",
              summarize_drift(expected, actual, state_fields(m), threshold=1e-5))
        write(output / "formal-native-substeps-receipt.json", guard.save(output))
        write(output / "joint-margin-monitor.json", joint.receipt())
        match = len(errors) == len(actions) and max(errors, default=0.) <= 1e-5
        out = dict(started_at=started, ended_at=stamp(), status="completed" if stop is None else "safety_stopped",
                   attempt=str(attempt), steps=len(errors), expected_steps=len(actions),
                   max_state_abs_error=max(errors, default=None),
                   first_state_error_gt_1e_5=next((i for i,x in enumerate(errors) if x>1e-5), None),
                   checker_success=bool(ref.env._check_success()),
                   terminal_success_streak_10=bool(len(success)>=10 and all(success[-10:])),
                   expected_checker_success=result["checker_success"], safety_stop=stop,
                   reproducible=bool(match and success and success[-1]==result["checker_success"] and
                       (tail is None or (tail_check is not None and tail_check["reproduced"]))),
                   partial_tail_replay=tail_check,
                   replay_kind="saved human actions; no per-step state injection or new human input",
                   replay_video=str(output / "replay.mp4") if video else None,
                   rendering_enabled=video, scientific_route_outcomes=0,
                   formal_train_ready=False)
        write(output / "result.json", out)
        print(json.dumps(out), flush=True)
    finally:
        for name in ("observation_renderer", "renderer"):
            obj = getattr(ref, name, None)
            if obj is not None:
                obj.close()
        ref.env.close()


def sweep(attempt, output, begin_interval=0, end_interval=None):
    from mobiwam.task_video_identity import source_model, sha
    from mobiwam.reference_collision import SweptGeometry
    from mobiwam.contact_rules import RULE_VERSION
    result, metadata, actions, states, integration, events = load_attempt(attempt)
    source = attempt.parent.parent
    m = source_model(str(source / "model.xml"), sha(source / "model.xml"))
    with np.load(attempt / "formal-native-substeps.npz") as z:
        q, phases, times, indices = z["qpos"], z["phases"].tolist(), z["sim_time"], z["step_index"]
    if len(q) != len(phases)+1 or len(indices) != len(phases) or len(q) != len(times):
        raise ValueError("Native trajectory alignment differs")
    if q.shape[1] != m.nq or any(p != phase_at(result, int(i)) for p,i in zip(phases, indices)):
        raise ValueError("Unexpected human native phase trajectory")
    geom = SweptGeometry(m, target_prefix=json.loads((source / "target-binding.json").read_text())["fixture_name"],
                         margin=.0005)
    minimum, leaves, checked = .1, 0, 0
    started = stamp()
    end_interval = len(phases) if end_interval is None else end_interval
    if not 0 <= begin_interval < end_interval <= len(phases):
        raise ValueError("Invalid interval partition")
    for begin in range(begin_interval, end_interval, 250):
        end = min(begin+250, end_interval)
        out = geom.path(q[begin:end+1], phases[begin:end])
        if not out["valid"]:
            segment = begin + out["segment"]
            out.update(segment=segment, original_control_step=int(indices[segment]),
                       simulated_time_interval=[float(times[segment]), float(times[segment+1])])
            if out.get("pair"):
                witnesses = []
                for at in (segment, segment+1):
                    pairs, distances = geom.distances(q[at], phases[segment])
                    matches=[i for i,pair in enumerate(pairs) if [geom.names[x] for x in pair]==out["pair"]]
                    witnesses.append(dict(native_state_index=at,
                                          distance_m=float(distances[matches[0]]) if matches else None))
                out["endpoint_pair_witnesses"] = witnesses
            checked = segment
            break
        checked = end
        minimum = min(minimum, out["lower_bound_m"])
        leaves += out["leaf_intervals"]
        write(output / "progress.json", dict(at=stamp(), checked_native_intervals=checked,
              total=len(phases), lower_bound_m=minimum))
        print("SWEPT_PROGRESS", checked, len(phases), minimum, flush=True)
    else:
        out = dict(valid=True, lower_bound_m=minimum, segments=checked, leaf_intervals=leaves)
    out.update(started_at=started, ended_at=stamp(), attempt=str(attempt), checked_native_intervals=checked,
               total_native_intervals=len(phases), begin_interval=begin_interval, end_interval=end_interval, evaluations=geom.evaluations,
               contact_rule_version=RULE_VERSION, required_clearance_m=.0005,
               new_physics_steps=0, scope="Original native substep qpos path with conservative joint-linear / rigid quaternion-geodesic interpolation; original contact rules; not a proof about unsampled physical paths",
               formal_train_ready=False)
    write(output / "result.json", out)
    print(json.dumps(out), flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode", choices=("replay","sweep"))
    parser.add_argument("--attempt",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--pilot",type=Path)
    parser.add_argument("--begin-interval", type=int, default=0)
    parser.add_argument("--end-interval", type=int)
    parser.add_argument("--video",action="store_true",help="Optional replay rendering")
    parser.add_argument("--cpu-only",action="store_true",help="Require masked CUDA and no video renderer")
    args=parser.parse_args()
    if args.cpu_only and (args.video or os.environ.get("CUDA_VISIBLE_DEVICES") != ""):
        raise ValueError("CPU-only mode requires empty CUDA_VISIBLE_DEVICES and no video")
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output / "process.json",dict(started_at=stamp(),pid=os.getpid(),command=sys.argv,
          python=sys.executable,cpu_only=args.cpu_only,CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
          MUJOCO_GL=os.environ.get("MUJOCO_GL")))
    try:
        if args.mode=="replay":
            if args.pilot is None:
                raise ValueError("Replay needs exact pilot config")
            replay(args.attempt.resolve(),args.output.resolve(),args.pilot,args.video)
        else:
            sweep(args.attempt.resolve(),args.output.resolve(),args.begin_interval,args.end_interval)
    except Exception as exc:
        write(args.output / "mechanical-error.json",dict(at=stamp(),type=type(exc).__name__,detail=str(exc)))
        raise


if __name__=="__main__":
    main()
