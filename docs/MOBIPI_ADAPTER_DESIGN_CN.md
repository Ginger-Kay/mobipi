# Mobi-pi E/D/A 适配器实现设计

## 已由固定源码和实际 checkpoint 确认的事实

- 主仓库：Mobi-pi `release`，服务器 commit `19b130b8...`；
- `CloseSingleDoor` policy 是 robomimic `RolloutPolicy` 包装的 `BC_Transformer`；
- checkpoint：`frame_stack=10`、`context_length=10`、`supervise_all_steps=true`、
  `pred_future_acs=true`；
- 环境动作是 12-D，Mobi-pi navigation 把 base command 写入 `action[7:10]`；
- `FrameStackWrapper` 会把前一步完整 action 写入 observation 的 `actions` 字段；
- upstream docking 后使用 `reset_before_rollout=False`，其 rollout helper 会读取现有 frame history，
  再调用 `RolloutPolicy.start_episode(lang)` 重置 policy；
- `EnvRobosuite.get_state()` 只返回 model XML、flattened MuJoCo state 和 episode metadata，
  `reset_to()` 不承诺保存 controller integrator、wrapper history 或 RNG。

12-D action 的固定源码语义已经确认：

```text
0:3    right-arm normalized delta position, base/controller-origin frame, scaled to +/-0.05 m
3:6    right-arm normalized delta axis-angle, same frame, scaled to +/-0.5 rad
6      torso command
7:10   base [forward, lateral, yaw] command; controller internally swaps the first two axes
10     gripper
11     mode (>0 desired-goal update, <=0 achieved-pose update)
```

对应的纯数学实现位于 `src/mobiwam/mobipi_actions.py`，接口标识为
`pandaomron-hybrid-mobile-v1`。

## 为什么仍能得到 10-step nominal chunk

公开 `RolloutPolicy.__call__` 最终只返回 `BC_Transformer` 输出的第一个 future action，但底层
network 在该 checkpoint 上输出长度为 10 的 sequence。适配器应复用 wrapper 的 observation
prepare、action unnormalization 和 rotation conversion，而不是复制一个近似版本。

硬门：同一 observation、同一 policy reset 下，暴露的 `chunk[0]` 必须与官方
`RolloutPolicy(ob)` 在 `1e-6` 容差内一致；否则 E/A paired 比较无效。

## paired source snapshot v0

首批 source 只允许在 episode reset 后、尚未接触物体时捕获。保存：

1. `env.get_state()` 的 XML、sim state 和 episode metadata；
2. FrameStackWrapper 的 10-frame history 和 timestep；
3. Python、NumPy、Torch CPU/CUDA RNG state；
4. environment seed、placement RNG state、policy/checkpoint/code hashes；
5. restore 后第一帧 observation hash。

每条路线前调用 `reset_to`，恢复 wrapper/RNG，再验证 state 与 observation hash。v0 利用
reset-boundary controller state 是 canonical 这一限制；不得宣称支持任意 mid-contact restore。

## 三条 executable route

### E: Execute

在 source history 上 reset policy，暴露 10-step nominal chunk，锁定 `action[7:10]=0`，执行
该 macro。每个控制步后显式恢复 PandaOmron 的 forward / side / yaw 三个平面底盘关节并清零
对应 qvel，再刷新 frame stack 的最新 observation；因此 E 的底盘保持固定，冻结 policy 可继续
执行到任务结束，用于生成 terminal labels。

### D: Dock

从同一 source restore，执行一个登记过的 collision-free base path。到达后持续发送 12-D zero
action，直到线/角速度低于阈值，并至少执行 10 steps，使 frame stack 只含稳定 post-dock
observations 和 zero base actions。随后调用 `start_episode`，从真实 post-dock context 产生新
chunk。绝不能调用 `env.reset()`。

第一版 canonical D 可回到训练 default base pose，用于验证 route diversity；论文版再增加
Mobi-pi scorer 产生的 D candidates，并把 candidate budget 与其它方法对齐。

### A: Assist

从同一 source restore，使用与 E 完全相同的 10-step nominal chunk。E rollout 在每一步读取
真实 achieved EE pose，并把 nominal OSC delta 转成 world-frame desired EE pose；A 再给定短
base trajectory，逐步计算
`T_BE_des = inverse(T_WB_assist) * T_WE_des`，由实际 controller/IK 生成 arm compensation，
同时把 base command 写入 `action[7:10]`。不要直接在 action 数组上做未经坐标验证的相减。

首个 canonical A 不另起一个轨迹网络：复用 D 的目标 pose，只执行朝 D 方向的 25% 前缀，
并截断到 5 cm / 3 deg / 10 steps。每一步用“预计下一时刻 controller origin”表达 world-frame
target，再反解 OSC delta。当前 PandaOmron 直接使用已有 OSC_POSE，不需要 DreamTrajectory，
也不需要额外 IK；后续双臂 embodiment port 才需要双臂 IK/闭链约束。

硬门是 `A(alpha=0)==E`：动作、state trace、observation 和 outcome 都应在登记容差内一致。

## 3-state smoke 的顺序

1. snapshot restore：3 个 source 各 restore 3 次；
2. future chunk exposure：`chunk[0] == official action`；
3. E：使用 MuJoCo planar-joint lock 保持 source base pose，最大位移严格小于 1 mm；
4. D：navigation 后连续 10 个 zero-action settle frames，history 中不得残留 base command；
5. A(0)：与 E 一致；
6. A small：只试 2--5 cm 无接触 base path，检查 intent error、IK、collision；
7. `mobiwam-collect --source-count 3` 得到 exactly E=3/D=3/A=3；
8. 通过后才运行 100 个 source states。

## WAM、DreamTrajectory 与 RoboCasa365 的位置

- 这 300 条是工程 pilot，不足以从头训练可靠 WAM；先验证 route oracle 是否显著优于 best
  fixed route，再扩展 candidates、seeds 和 source states；
- DreamTrajectory-style 模型只评价给定 manipulation/action trajectory，不能覆盖 D 的可变长
  navigation、settle 和 post-dock history consequence，因此是 matched trajectory-only baseline；
- MobiWAM 学的是 typed option-level consequences，包括 success/risk/visibility/reachability/
  contact/progress/cost；
- RoboCasa365 v1.0.1 只在主栈通过后做 held-out task/scene 和 embodiment transfer，不参与
  Mobi-pi checkpoint 的原生 paired collection；其 2026-07 composite-data 逐帧 stage annotations
  可作为 stage encoder 的辅助监督，但不是 E/D/A route label。
