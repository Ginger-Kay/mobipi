# 100 个 source states 的 E/D/A paired 数据采集与打标

## 你需要先理解的一句话

不是分别随机采 100 条 E、100 条 D、100 条 A。正确做法是先固定 **100 个相同 source
states**，每个 state 恢复三次，分别执行 E、D、A，得到 100 组三元组、共 300 条 rollout。

```text
source state s_001
  +-- restore -> E -> labels/video/trace
  +-- restore -> D -> labels/video/trace
  +-- restore -> A -> labels/video/trace
```

`source_state_id` 是统计和 train/validation/test split 的最小单位。同一组三条路线永远在
同一个 split，不能把单条 rollout 随机打散。

当前 CloseSingleDoor checkpoint 已从服务器实际 `config.json` 确认：frame stack 为 10，
Transformer context length 为 10，且 `pred_future_acs=true`。公开 `RolloutPolicy` 只返回第一个
动作，但底层网络输出 10-step future-action chunk。适配器可暴露完整 chunk，但必须先证明反归一化
后的 `chunk[0]` 与原 wrapper 动作数值一致。E 和 A 共享这个同源 10-step chunk；D 在真实
post-dock 稳定观测上重新产生自己的 chunk。

## 三条路线和 X

- `E`：同一 frozen policy nominal chunk，底盘固定；
- `D`：navigate -> zero-action settle -> 用 10 个稳定 post-dock frames 冲掉 navigation history
  -> `RolloutPolicy.start_episode` -> re-query policy -> execute；这里绝不能 `env.reset()`，否则会撤销 docking；
- `A`：复用 E 的同一个 nominal chunk，把它转成 world-frame EE intent，再生成 base motion、
  arm compensation、IK 和 hard filtering；
- `X`：不是第四条 rollout。当 E/D/A 没有安全且成功的候选时，从结果派生 `X=true`。

因此 DreamTrajectory 不负责实现 A。它在后面读取 A 候选和 induced EE trajectory，作为
trajectory-only evaluator baseline。MobiWAM 才负责预测三条 route 的 option-boundary
consequences 并选择 E/D/A/X。

## 采集前硬门

以下任何一项没有通过，都不要开始 300 条数据：

1. vanilla rollout 可复现，checkpoint 和 HDF5 checksum 完整；
2. action semantics 已确认，base-locked replay 与 upstream 一致；
3. transform closure 通过；
4. 5 个 snapshots 各 restore 3 次，首帧 observation 与 terminal outcome 达标；
5. `A(0) == E`；
6. D 的 navigate-settle-history-reset-re-query 生命周期有完整日志；
7. 每条 rollout 都能保存视频、state trace 和 action trace。

服务器当前 Mobi-pi 代码只显示 `sim.get_state().flatten()` 等物理 state 读取入口，没有现成的
“physics + controller integrator + policy history + RNG”完整 paired snapshot。必须补 adapter，
不能把一个 MuJoCo qpos/qvel 向量误称为完整 snapshot。

## 100 个 state 怎么取

`paired_collection_v0` 是工程和打标 pilot，先全部取 reset-boundary precontact states，保证
controller 内部状态是 canonical，且 E/D/A 都合法。此时 snapshot 使用模型 XML、flattened
MuJoCo state、episode metadata、frame history、wrapper timestep 和 RNG state；不能把这个结论
外推到任意 mid-contact snapshot，后者还需要显式序列化 controller integrator/goal。
第一批 100 个 source states 使用可复现的分层采样，不靠人工挑视频：5 个 Mobi-pi evaluation
layouts（`1,4,7,8,9`），每个 layout 20 个 state；每个 layout 内分别用
`randomize_base_init_pose = 0, 0.03, 0.05, 0.10`，每档 5 个 environment seeds。这个参数会同时
以米和弧度扰动 `(x,y,yaw)`。只过滤 reset 即碰撞或仿真无效的 state，禁止根据 E/D/A 成败
事后删样本。

采完后再按自动几何标签检查是否覆盖下列 strata；下表是 coverage audit，不是人工配额：

| strata | 数量 | 目的 |
|---|---:|---|
| ordinary / E 足够 | 25 | 检查 selector 是否过度干预 |
| viewpoint mismatch | 20 | 检查 D 的价值 |
| reach/joint-margin 边界 | 20 | 检查 A 的价值 |
| occlusion/visibility challenge | 15 | 检查未来后果是否可预测 |
| mixed hard states | 20 | 形成 failure 和 X 标签 |

canonical `D` 使用登记过的 collision-free 路径回到 task default / Mobi-pi docking pose；canonical
`A` 朝同一个 D pose 只走 25% 路程，并截断到 5 cm、3 deg、10 steps。这样三个 route 有明确的
成本层次：已经对齐时 E 最便宜，小偏差时 A 可能足够，大偏差时 D 才值得执行。

第一任务可先用 CloseSingleDoor 打通；这 100 states 不能作为“跨 task-family”论文证据。论文级
数据随后至少增加一个不同 family，并扩成 D/A 多候选、多 seeds 的 route-oracle 实验。

## 每条自动标签

必须自动记录：

```text
source_state_id, task_id, task_family, episode_id, split, stage
route_type(E/D/A), candidate_id, candidate_params, repeat_index
environment_seed, policy_seed, route_seed
policy/checkpoint/simulator/code/snapshot/observation hashes
action_semantics_id, history_protocol_id
transform_check_passed, restore_check_passed
stage_eligible, hard_valid, invalid_reason
success, irreversible_failure, collision, contact_loss, failure_type
task_progress_before, task_progress_after, progress_delta
visibility, policy_compatibility, reachability, joint_margin
intent position/rotation error, execution time, base path, route cost
video_path, state_trace_path, action_trace_path
```

`success/progress/contact/collision` 可使用 simulator privileged state 生成监督标签，但部署时的
selector input 禁止读取这些 privileged fields。人工只复核 failure taxonomy 和自动标签冲突，
不要手工删除失败或 invalid branch。

## 目录和校验

```text
experiments/mobipi/paired_eda_v0/
  manifests/source_states.jsonl
  manifests/route_rollouts.jsonl
  snapshots/<source_state_id>/...
  rollouts/<source_state_id>/<route_type>/...
  reports/paired_validation.json
```

不要让 rollout 进程直接向两个 JSONL 追加。`mobiwam-collect` 先把每个 source state 的完整
`E/D/A`（以及同一 snapshot 的全部 repeats）原子提交到 `transactions/source-XXXXXX.json`，
再重建 manifests。碰撞、饱和、dock timeout 和 settle timeout 是需要保留的 route 结果，写成
`hard_valid=false`；只有进程崩溃或接口异常才会阻止该 source 提交。重跑时会跳过已经完整
提交的 source index。

本地已经实现 `mobiwam.adapters.mobipi:create_adapter`；它必须先在服务器真实 checkpoint /
CLIP cache / RoboCasa assets 上通过下面的 `A(0)==E` 和 3-source smoke，当前不能把“代码已实现”表述成
“仿真已验证”。先运行：

```bash
cd /share/chensiyu/MobiWAM
PYTHONNOUSERSITE=1 envs/mobipi/bin/python scripts/verify_mobipi_a_zero.py \
  --config configs/mobipi_close_single_door.json \
  --output-root experiments/mobipi/a_zero_gate
```

再跑 3 个 source states：

```bash
cd /share/chensiyu/MobiWAM
PYTHONNOUSERSITE=1 envs/mobipi/bin/mobiwam-collect \
  --adapter-factory mobiwam.adapters.mobipi:create_adapter \
  --adapter-config configs/mobipi_close_single_door.json \
  --output-root experiments/mobipi/paired_eda_smoke \
  --source-count 3 \
  --repeats-per-route 1
```

只有 smoke 得到 exactly `E=3, D=3, A=3`，并通过视频、trace、restore、transform 与
`A(0)==E` 检查后，才把 `--source-count` 改为 100。在 vanilla rollout、snapshot restore、
future chunk 和 transform contract 通过前，命令必须失败，而不是降级生成伪数据。

采集后执行：

```bash
cd /share/chensiyu/MobiWAM
PYTHONNOUSERSITE=1 envs/mobiwam/bin/mobiwam-dataset \
  --sources experiments/mobipi/paired_eda_v0/manifests/source_states.jsonl \
  --rollouts experiments/mobipi/paired_eda_v0/manifests/route_rollouts.jsonl \
  --report experiments/mobipi/paired_eda_v0/reports/paired_validation.json \
  --expected-source-states 100 \
  --repeats-per-route 1
```

通过条件是 exactly `E=100, D=100, A=100`、无 missing/duplicate branch、无 split 泄漏、
所有 restore/transform checks 通过。`X` 数量由 300 条结果派生并写入 report。

## 这 300 条之后做什么

1. 先画每个 state 的 E/D/A outcome，不训练 WAM；
2. 计算 canonical-route oracle 与 best fixed route 的 gap；
3. 若最优 route 几乎恒定，先改任务分布或 D/A generator；
4. 若 route diversity 存在，扩 D/A 候选与 3 个 seeds；
5. 再训练 geometry/rule 和 value-only；
6. 再接 DreamTrajectory-style trajectory-only；
7. 最后训练 MobiWAM structured consequence evaluator；
8. 只有 structured evaluator 仍解决不了关键 3D/时序歧义时，增加 4D/video auxiliary。
