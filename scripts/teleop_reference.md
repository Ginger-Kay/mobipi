# Human teleoperation references

Created: 2026-09-08. Target: pinned RoboCasa 0.2 fork, PandaOmron,
Python 3.10.20 / MuJoCo 3.2.6. This is human reference tooling, excluded
from formal OBC training/validation/test. It is not the autonomous planner.

The `teleop_reference.py` entry combines OSC arm and base commands in one
12-dimensional `env.step` action. Arrows and arm rotation keys follow the
fork's mirrored keyboard convention; W/A/S/D independently command base
translation, Z/X yaw, Space toggles the gripper. Opposite keys cancel.
Mouse operations are for observing the scene, not applying physical perturbations.

The panel provides practice, pause, save source, record, finish, contact/dock
markers, route selection, restore, and replay. Equivalent shortcuts are
F1/F2/F3/F4/F5/F6/F8/F9; Esc pauses/resumes. Browser shortcuts may intercept
function keys, so the panel is preferred. The application must run on its own
dedicated X desktop: it grabs keyboard delivery while pynput observes input,
preventing MuJoCo display shortcuts from also reacting to control keys.
Mouse camera navigation and panel buttons remain available. The grab is
released when the application exits. No recording occurs in practice.

## Operator sequence

1. Practice simultaneous base/arm control. Select an unobstructed development
   pose with an open target and gripper not contacting it. Inspect the target
   fixture and full-robot view. Saved sources are marked pending contact and
   collision review; the tool does not certify a valid source automatically.
2. Save source. Start with route A; Record restores that exact model/state with
   canonical controller initialization. Do not save a new source merely to
   make E/D succeed. Practice is not an experiment outcome.
3. Approach, make stable finger/handle contact, mark Contact, then move base and
   arm together while closing. Avoid closing by arm/body collisions. Stop saves
   failed or unfinished attempts as well. Ten successive official checker
   successes automatically finish recording; this alone does not qualify A.
4. Replay last attempt. It restores the recorded source and feeds actions with
   no keyboard listener influence; the live viewer is detached during replay
   so mouse perturbations cannot influence physics. A view-only preview shows
   the rendered frames and step progress. Replay video is saved separately.
5. Inspect original/replay, contact traces and base/arm/target progression.
   Only after review label a successful A reference. Prioritize Drawer, then
   Door, then same-source E/D. E commands zero base velocity; D does so after
   the Dock marker. These are controller commands, not welded physical joints;
   actual drift must be measured against the 2 cm criterion.

## Artifacts and limits

Every source has model XML, full MuJoCo integration state, episode metadata,
RNG state and an initial trace. Restoration reloads XML, installs state,
refreshes controller origins/state and initializes OSC achieved goals in its
configured frame. No hidden physics step precedes recording.

Every attempt gets HDF5 actions/states, initial integration state, JSONL
pre/post-step traces (qpos/qvel, arm joints, base pose, target joints, contacts,
input keys, timestamp and camera), result/events, and original.mp4. States have
length actions+1. Video frame t follows action t. Source XML uses existing
asset paths; archive assets separately before moving to another machine.
Replay saves its own MP4, per-step state errors, earliest error >1e-5 and
checker result. Exact replay is a diagnostic, not scientific source identity.

Recorded frame rate is 20 simulation Hz, default 1920x1080. Wall-clock speed
depends on hardware; trace wall times distinguish slow execution from realtime.
Policy observations are reconstructible from recorded state/model/config;
this entry does not directly export a policy-ready observation dataset.
Contact pairs are recorded without automatically classifying allowed contacts.
Neither collision safety, same-source fairness nor the 5 cm / 40% / 80% A
criteria is inferred merely from task success. Those remain post-record review.

The fork's base controller swaps its first two input axes, transforms using
base yaw, and scales by actuator ranges. Its slide friction is 250 N with
kv=1000, so raw .06 commands stall. Translation keys send normalized .35,
yaw .20; no friction, gain, robot or physical parameter is changed. Arm delta
keys send .06 normalized (3 mm position or .03 rad rotation per control step).
The OSC reset_goal implementation stores world coordinates even with a base
input frame; restoration explicitly calls zero-delta achieved set_goal before
allowing moving-base desired-goal accumulation.

## Launch / resume

Use a dedicated private tmux session with logs, record provenance and perform
the GPU allocation audit first. Example environment for the verified desktop:

```bash
cd /share/jhk/MobiWAM/Mobipi
DISPLAY=:29 PYTHONNOUSERSITE=1 LD_LIBRARY_PATH=/share/jhk/MobiWAM/env/lib \
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=3 CUDA_VISIBLE_DEVICES=3 \
XDG_CACHE_HOME=/share/jhk/MobiWAM/cache/teleop \
/share/jhk/MobiWAM/env/bin/python -u scripts/teleop_reference.py \
  --output /share/jhk/MobiWAM/artifacts/MMWAM-OBC-002/teleop-reference/UNIQUE-RUN
```

Add `--source /absolute/source-directory` to resume a saved source, or
`--replay-attempt /absolute/source/A/attempt-directory` for replay without a
keyboard device. Saved source env_config overrides task/layout/controller
defaults. For a new Door run use `--task CloseSingleDoor`, then inspect the
selected target fixture before freezing a source.

Use `--resume-attempt /absolute/source/A/attempt-directory` to reopen the
interactive interface paused at that source, with F9 pointing to the saved
attempt. Existing recordings remain untouched.

`--self-test` is an explicitly labelled engineering diagnostic: short mixed
commands, actual base/arm movement, HDF5/video and action replay. It does not
produce a human success reference. Historical diagnostic runs r1/r2/r3 used
less strict movement checks; r4 adds a >5 mm base-response threshold after
the stiction diagnosis. Keep those histories rather than reporting all as
equivalent runtime qualifications.
