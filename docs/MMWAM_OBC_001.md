# MMWAM-OBC-001 execution notes

This branch is the independent A800 implementation for the OBC-WAM sprint. The upstream Mobi-pi base is `19b130b8ada3f7e029918449c12d433e9e629ca1`; the imported RoboCasa compatibility layer is `426bc4dbbadec923d37752b012ba1152d25f8716`. Donor outputs are reference evidence only.

Runtime roots are fixed:

- code: `/share/jhk/MobiWAM/Mobipi`
- environment: `/share/jhk/MobiWAM/env`
- data/checkpoints: `/share/jhk/MobiWAM/data`, `/share/jhk/MobiWAM/checkpoints`
- run artifacts: `/share/jhk/MobiWAM/artifacts/MMWAM-OBC-001/runs`
- caches: `/share/jhk/MobiWAM/cache/{conda,pip,huggingface,torch}`

Every Python command uses `PYTHONNOUSERSITE=1`. A tracked simulator config contains `code_commit: BIND_AT_RUN`; `mobiwam-bind-config` refuses a dirty tree and creates a non-overwriting run config bound to the exact HEAD. Formal GPU commands run only from that config and a run manifest.

## Typed interface

- E: `QUERY -> EXECUTE -> REPLAN/terminal`, with a hard planar base lock.
- D: `MOVE -> SETTLE -> OBSERVE -> RESET -> QUERY -> POST_DOCK_POLICY_READY -> EXECUTE -> REPLAN/terminal`.
- A: one policy query followed by simultaneous `EXECUTE || ASSIST`, with world-frame EE intent compensation.
- X: derived `SAFE_EXIT`; it is never collected as a simulator rollout arm.

Snapshots include MuJoCo state/model metadata, observation history, Python/NumPy/Torch/CUDA/environment RNG, numeric controller buffers, and contact/controller fingerprints. Every route restores and verifies the source before execution. Formal records default to no video; state/action/event traces remain mandatory, and audit media must be enabled only for a pre-registered subset.

## Frozen learned ladder

Under C2, `configs/obc_wam_v1.yaml` fixes the compact primary profile: a shared frozen encoder, four `d=256` Transformer layers, eight heads, `ff=2560`, typed event/phase/duration tokens, E/D/A low-rank adapters, and 5–10M trainable parameters. `value-only`, `trajectory-only`, and `obc-wam` use the same backbone capacity and source/candidate features and remain within 10% trainable parameters. Observable tensors and simulator-only labels are written to separate artifacts; model forward paths open only the observable partition. Training seeds are 17/23/41; validation selects epochs, calibration freezes temperatures/operating points, and the locked evaluator refuses non-`locked_test` rows.

## Safety boundary

This branch contains no GPU lease, preemption, kill, stop/restore, or donor-runtime wrapper. GPU selection is performed outside the model code from a recorded 10-second `nvidia-smi` series. Unknown processes are never modified. Each logical unit gets at most five lineage-preserving attempts, a unique artifact root per attempt, and tmux run/status windows for long jobs.

The endpoint retention process is `/share/chensiyu/pi0.5_test.py`. Platform cleanup may occur after more than 60 minutes at `<=1%` chip utilization. Never kill, pause, modify, or duplicate an existing retention process. If it is absent and every allocation-visible GPU has reached `0%` utilization, restore one instance from the researcher-approved path; verify it again before handoff. Its utilization is infrastructure retention, not MMWAM runtime evidence.
