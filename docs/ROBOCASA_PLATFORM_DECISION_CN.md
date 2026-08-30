# RoboCasa365 与 Mobi-pi RoboCasa 0.2.0 平台决策

## 结论

这里的“最新 RoboCasa”明确指 **RoboCasa365 `v1.0.1`**。主线 Gate 0 和首批 paired
`E/D/A` 数据仍使用 **Mobi-pi 内置 RoboCasa 0.2.0**；RoboCasa365 建立为完全隔离的第二环境。
它不是可有可无的附录，而是在原生栈通过 `A(0)==E` 和 3-source smoke 后立即启动的 port
track，用于 held-out task/scene transfer 和 embodiment audit。

这不是因为 RoboCasa365 不够新，而是因为当前实验首先要回答“同一个冻结策略在同一个
source state 应选 E、D 还是 A”。如果同时更换仿真器、机器人、controller、observation、
action space 和 checkpoint，实验失败时无法判断是 route 方法无效，还是迁移接口错误。

## 已核对的版本事实

| 项目 | Mobi-pi bundled stack | RoboCasa365 |
|---|---|---|
| RoboCasa | `0.2.0`，commit `3683fb0...` | `1.0.1` |
| Python | `3.10.20` | 官方安装说明为 `3.11` |
| robosuite | `1.5.0` | 官方要求 clone `master` |
| MuJoCo | `3.2.6` | 随当前 robosuite/环境解析 |
| API | Mobi-pi/robomimic evaluator | Gymnasium `gym.make(...)` |
| 机器人 | Mobi-pi task 把 `PandaMobile` 映射为单臂 `PandaOmron` | 代码包含 PandaMobile、GR1、GR1FixedLowerBody、G1 等配置 |
| policy | 官方 Mobi-pi checkpoint 原生匹配 | 没有 Mobi-pi checkpoint 兼容保证 |
| 资产 | bundled release 的旧资产协议 | 约 10 GB 新资产；365 tasks、2500+ scenes、3200+ objects |
| 数据标签 | Mobi-pi HDF5 的 task/action/observation | 2026-07 更新的 composite datasets 含逐帧 subtask、atomic skill、pick/place/navigate stage 和语言指令 |

来源：RoboCasa365 官方 README、RoboCasa 1.0.1 文档、arXiv:2603.04356，以及服务器
`/share/chensiyu/MobiWAM/repos/mobipi` 的固定源码和实际环境。RoboCasa365 是对 RoboCasa
的扩展，但不是对 Mobi-pi 运行栈的 in-place drop-in upgrade。

## 为什么主线先用 0.2.0

1. checkpoint、训练 HDF5、env metadata、normalization stats 和 evaluator 原生一致；
2. `CloseSingleDoor` vanilla 路径已经固定，能最短闭合 E 的证据链；
3. Mobi-pi 已提供 docking/navigation 代码，`action[7:10]` 是现成 base command slice；
4. paired branching 最敏感的是 snapshot、controller state 和 policy history，先减少变量；
5. 论文的主创新是 typed route selection 和 consequence prediction，不是发布新仿真器。

## RoboCasa365 的正确用途与进入条件

RoboCasa365 的价值是任务、场景和 embodiment 多样性。原生栈通过最低运行门后，第二环境做：

- held-out kitchens 和新的 task strata；
- 检查 selector 是否依赖 RoboCasa 0.2 的视觉或布局偏差；
- 选择 GR1FixedLowerBody 等双臂配置做 embodiment adapter 预研；
- 验证 `E/D/A/X` schema、stage gate 和 consequence heads 是否仍成立。

其中逐帧 stage annotation 对我们的 **stage gate** 很有用，但必须区分两类标签：

- RoboCasa365 的 `pick/place/navigate` 是行为阶段，可用于预训练或辅助监督 stage encoder；
- 我们的 `E/D/A/X` 是 route 决策标签，只能由同一 source state 的反事实 rollout 结果产生。

不能把 `navigate` 直接当成 `D`，也不能把 `pick/place` 直接当成 `E/A`。正确做法是先把
RoboCasa365 annotation 映射到统一的 precontact/contact/transition 辅助字段，再在 Mobi-pi 的
paired 数据上学习 route consequence。

它不能直接复用 Mobi-pi checkpoint 作为“同一个 policy”的对照。进入第二环境时必须重新登记
policy/checkpoint、action semantics、camera、controller、snapshot contract 和 task success。

port 按四个门执行，不能直接采 100 组：

1. `RC365-P0`：独立 Python 3.11 环境创建同名 `CloseSingleDoor`，确认 observation keys、
   PandaMobile/PandaOmron action dimension 和 controller semantics；
2. `RC365-P1`：加载冻结 Mobi-pi checkpoint 做 1 回合 compatibility smoke。能运行不等于
   policy 有效，必须单独记录 success 和分布偏移；
3. `RC365-P2`：若接口兼容，验证 snapshot restore 与 `A(0)==E`；若不兼容，停止复用
   checkpoint，改接 RoboCasa365 官方 policy stack，并登记新的 policy contract；
4. `RC365-P3`：先采 20 个 paired source states。只有 E/D/A 都能真实分叉且 route preference
   非恒定，才扩到 100 组和 held-out task/scene。

因此当前 100 组不是在两个平台各盲采一次。先用原生栈生成可信标签协议，再让 RoboCasa365
验证该协议能否迁移；这比同时换 simulator、policy 和 route controller 更容易形成可审计的
论文证据。

## 两环境目录

```text
/share/chensiyu/MobiWAM/
  envs/mobipi/                 # Python 3.10，冻结主环境
  repos/mobipi/                # Mobi-pi + bundled RoboCasa 0.2
  envs/robocasa365/            # Python 3.11，独立环境
  repos/robocasa365/           # 固定 RoboCasa365 tag/commit
  experiments/mobipi/          # 主 paired 数据与 route oracle
  experiments/robocasa365/     # 后续 transfer，不混写数据
```

禁止把 RoboCasa365 或 robosuite master 安装进 `envs/mobipi`，也禁止让两个环境共用可写的
RoboCasa asset/config 目录。

今天固定的 port 版本是 RoboCasa commit `a07e365c...` 和 robosuite master commit
`5ce6643f...`。部署命令为：

```bash
cd /share/chensiyu/MobiWAM
bash scripts/bootstrap_robocasa365_env.sh
tmux new-session -d -s mobiwam-rc365-assets \
  'cd /share/chensiyu/MobiWAM && bash scripts/download_robocasa365_assets.sh'
```

资产门通过后先运行 `scripts/probe_robocasa365_interface.py`，只生成接口审计 JSON，不采集
E/D/A。探针确认 action/observation/controller/snapshot 接口后，才实现 RC365 adapter。

## 当前真正的 embodiment 风险

Mobi-pi 官方仿真任务是单臂 PandaOmron，而爱宝是真机轮式双臂。首月可用单臂 active-arm
协议验证 route selection，但论文和代码必须明确写成：

- minimum protocol：one active arm；
- bimanual rigid-grasp/contact constraint：extension；
- 真机前重新实现 `policy action -> dual-arm/base command` adapter；
- 不把 Panda 单臂仿真结果表述为爱宝双臂控制已经验证。

RoboCasa365 的 GR1 配置能帮助检查双臂 observation/action 接口，但不能替代爱宝真机控制器
和同步调度的验证。
