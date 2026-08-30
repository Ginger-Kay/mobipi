# MobiWAM Gate 0 Engineering Scaffold

This repository is the implementation shell for the ICRA 2027 project. It is
deliberately smaller than the paper idea: it first proves that the simulator,
frozen policy, route candidates, snapshots, and logging are trustworthy.

Chinese step-by-step guide: `docs/START_HERE_CN.md`.

## What the project is based on

| Component | Engineering role |
|---|---|
| Mobi-pi / the existing RoboCasa rollout stack | Primary simulator, frozen manipulation policy, vanilla execution, and docking candidates |
| RoboCasa365 v1.0.1 | Isolated held-out task/scene transfer stack after Gate 0; never an in-place Mobi-pi upgrade |
| DreamTrajectory | Trajectory-only evaluator and candidate-ranking baseline; it does not implement route `A` |
| Kinematic feasibility / reactive whole-body control | Reference for online base assistance while preserving end-effector intent |
| MobiWAM | Our typed option-boundary consequence evaluator after route diversity is proven |
| Video / 4D-WAM / DECOWAM | Optional auxiliary representations after matched value and trajectory controls |

The contribution is not "put a WAM on Mobi-pi." The testable contribution is a
stage-aware decision among:

- `E`: execute the frozen policy as-is;
- `D(p)`: dock, flush the 10-frame stack with stable zero-action observations,
  reset only the policy state, re-query the same policy, then execute;
- `A(alpha)`: move the base online while compensating the arm to preserve the
  frozen policy's world-frame end-effector intent;
- `X`: abstain or enter a safe state when no candidate is sufficient.

## Repository boundary

The deployed research snapshot is pinned in
`docs/RESEARCH_REPO_MANIFEST.md`. The primary simulator and policy repository
is now the official Mobi-pi `release` checkout at `repos/mobipi`, pinned to
`19b130b8ada3f7e029918449c12d433e9e629ca1`. It is intentionally kept separate
from this control scaffold and from all checkpoints, datasets, caches, and
rollout outputs.

This scaffold contains contracts, route records, analysis, tests, and runtime
guards. Simulator-specific adapters belong here only after the real rollout
repository and its exact command are identified. Do not guess class names or
action semantics from a paper.

Server layout:

```text
/share/chensiyu/MobiWAM/
  audit/          # machine, dependency, and GPU-lease evidence
  configs/        # interface contract and experiment configuration
  envs/           # isolated Conda environment
  experiments/    # manifests, rollouts, metrics, reports, figures
  logs/            # project logs only
  repos/           # research-control and upstream repositories
  scripts/         # operational wrappers
  src/             # this Python package and later simulator adapters
```

Only the first task is downloaded initially: `CloseSingleDoor` checkpoint seed
1. The verified checkpoint contains its environment metadata, shape metadata,
and action normalization statistics, so checkpoint-only evaluation does not
require the 9.60 GB source HDF5. Keep that HDF5 download optional for upstream
parity and retraining. Do not download all five tasks during bootstrap.

## First commands

Create the dedicated Mobi-pi environment on the server:

```bash
cd /share/chensiyu/MobiWAM
bash scripts/bootstrap_mobipi_env.sh
```

Then follow the artifact and CPU-ready gates in
`docs/MOBIPI_BOOTSTRAP_CN.md`. The smaller Gate 0 package remains in the
separate `envs/mobiwam` environment.

```bash
/share/chensiyu/MobiWAM/envs/mobiwam/bin/python -m pip install -e .
/share/chensiyu/MobiWAM/envs/mobiwam/bin/python -m unittest discover -s tests -v
```

Do not call a GPU command directly. After the final wrapper is synchronized and
its fake-process regression passes, use:

```bash
MOBIWAM_GPU_INDEX=0 bash scripts/gpu_with_lease.sh -- \
  bash scripts/run_mobipi_vanilla_once.sh
```

The occupant observed on 2026-08-26 is a multiprocessing tree with parent PID
`2402369`, one resource tracker, and four workers. The wrapper records the full
tree, sends `SIGTERM` to the parent and remaining descendants, refuses
`SIGKILL`, and runs the requested command only after every recorded process has
exited. On exit it restores the original executable, argument vector, working
directory, and a non-secret runtime-environment whitelist, then verifies that
the restored descendant count matches the original count.

The wrapper targets only:

```text
/share/chensiyu/CoTTA/streamingqa/scripts/pi0.5_test.py
```

The full environment is never copied into the audit directory because it may
contain credentials. Only Python, Conda, CUDA, PATH, dynamic-library, and thread
count variables needed for runtime reconstruction are retained.

The CPU-only regression harness substitutes one parent, one resource tracker,
and four spawn workers. It never targets the real occupancy command:

```bash
bash scripts/test_gpu_with_lease.sh
```

The tree-control version passed this harness on 2026-08-26 and left the real
occupancy tree unchanged. A later hardening change restricted saved environment
variables to a non-secret whitelist; that exact build must be synchronized and
the same harness rerun before the wrapper is used with the real occupant.

## Gate 0 order

1. Fill every blocking `UNKNOWN` in `configs/interface_contract.yaml` from the
   real code and one vanilla rollout.
2. Reproduce the vanilla policy and save observation, action, state, timestamps,
   video, checkpoint hash, and code commit.
3. Pass deterministic policy query, action replay, transform closure, snapshot
   restore, and synchronized base-arm dispatch tests.
4. Prove `A(0) == E`, then test three small non-contact assistance paths.
5. Complete the full `D(p)` lifecycle: save, navigate, settle, rebuild history,
   re-query, execute.
6. Run a 3-source paired smoke through `mobiwam-collect`. It commits a source
   only after the complete E/D/A tuple succeeds, so a crash cannot leave a
   partial source in the manifests.
7. Collect the engineering pilot from 100 source states with one canonical
   `E`, `D`, and `A` branch each: exactly 300 paired rollouts. `X` is derived.
8. Expand to 25 paired source states with `1 E + 3 D + 3 A`, each under three
   seeds, before the publication-scale route-oracle study.
9. Train no WAM until route preference is non-constant and the route oracle is
   meaningfully better than the best fixed route.

Source-verified runtime primitives now live in:

- `mobiwam.mobipi_actions`: exact PandaOmron 12-D layout and SE(3) world-intent compensation;
- `mobiwam.mobipi_policy`: 10-step chunk exposure through the original RolloutPolicy conversion path;
- `mobiwam.dock_protocol`: stable zero-action history flush before policy-only reset;
- `mobiwam.assist_trajectory`: canonical 5 cm / 3 deg short prefix toward the registered D pose.

The real adapter entry point is `mobiwam.adapters.mobipi:create_adapter`, with
its pinned configuration in `configs/mobipi_close_single_door.json`. It is
implemented and CPU-unit-tested locally, but remains **server-unverified** until
the downloaded checkpoint, CLIP cache, and RoboCasa assets pass `A(0)==E` and the
3-source smoke. `scripts/verify_mobipi_a_zero.py` is the first runtime gate.
The adapter loads environment metadata, shape metadata, and action normalization
statistics directly from the verified checkpoint, so the 9.60 GB source HDF5
is no longer a blocker for evaluation; it remains useful for retraining and a
later normalization-stat cross-check.

Platform rationale: `docs/ROBOCASA_PLATFORM_DECISION_CN.md`. Paired collection
and labeling guide: `docs/PAIRED_EDA_COLLECTION_CN.md`. The source-verified
adapter protocol is in `docs/MOBIPI_ADAPTER_DESIGN_CN.md`.

## Go / no-go rule for WAM

WAM remains a contribution only if a matched WAM evaluator improves route
selection over value-only and trajectory-only evaluators at the same candidate
budget, observation budget, and compute budget. Otherwise the paper should keep
the route-selection contribution and remove WAM from the main claim.
