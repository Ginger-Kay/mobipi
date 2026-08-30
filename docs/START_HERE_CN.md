# MobiWAM 从零实现与服务器操作手册

## 1. 先理解项目到底基于什么

当前项目不是把四篇论文的代码简单拼起来。各部分职责如下：

| 来源 | 在工程里的作用 | 是否为主代码库 |
|---|---|---|
| Mobi-pi / 现有 RoboCasa rollout | 仿真、冻结策略、vanilla rollout、移动操作任务 | 是 |
| RoboCasa365 v1.0.1 | Gate 0 后的 held-out task/scene transfer；独立环境 | 否 |
| DreamTrajectory | 只看候选轨迹的打分 baseline；不负责实现 A | 否 |
| Whole-body kinematic feasibility | 在线底盘辅助与机械臂补偿的实现参考 | 否 |
| MobiWAM | 我们自己的 typed option-consequence evaluator | 否 |
| Video / 4D-WAM / DECOWAM | matched baseline 之后的可选辅助表征 | 否 |

论文真正要验证的是：同一个冻结策略在某个决策时刻可能需要不同的执行路线，
系统能否根据阶段和预测后果，在以下四种路线间做正确选择：

- `E`：不移动底盘，直接执行冻结策略；
- `D(p)`：移动到候选 docking pose `p`，稳定后清空视觉历史，再询问同一策略；
- `A(alpha)`：执行策略的同时移动底盘，并补偿机械臂以保持世界系末端意图；
- `X`：没有安全候选时拒绝执行或进入安全状态。

WAM 的价值是预测这些候选执行后会发生什么，而不是替代 Mobi-pi，也不是先做一个大模型
再寻找用途。只有 route oracle 证明不同状态确实偏好不同路线之后，训练 WAM 才有意义。

## 2. 服务器目录

只在用户授权的目录中写入：

```text
/share/chensiyu/MobiWAM/
  audit/          机器、依赖、版本和 GPU 租约证据
  configs/        接口合同与实验配置
  envs/mobiwam/   本项目独立 Python 环境
  experiments/    rollout、指标、报告和图表
  logs/           本项目日志
  repos/           MM-WAM-Research 与真实工程仓库
  scripts/        环境、测试和 GPU 安全脚本
  src/            MobiWAM 的选择与适配代码
  tests/          CPU 单元测试和假进程测试
```

`MM-WAM-Research` 只保存研究依据和实验契约。Mobi-pi、RoboCasa、checkpoint、数据集和
大规模视频必须放在平行的工程目录，不能塞进研究仓库。

## 3. 第一次配置

### 3.1 创建最小环境

以下脚本使用 `conda-forge`，不触碰 Anaconda 默认 channel 的 ToS：

```bash
cd /share/chensiyu/MobiWAM
bash scripts/bootstrap_env.sh
```

验收：

- `/share/chensiyu/MobiWAM/envs/mobiwam/bin/python` 存在；
- `logs/bootstrap_env.log` 末尾出现 `Environment ready`；
- `audit/mobiwam_env.txt` 记录 Python 和所有包版本；
- 两个 Route Oracle 单元测试通过。

这个最小环境只运行 schema、collector 和 route-oracle 工具。真实 Mobi-pi rollout 使用已经
固定的 `envs/mobipi`，不把 MuJoCo/RoboCasa 依赖混装到此环境。

### 3.2 研究仓库

目标路径：

```text
/share/chensiyu/MobiWAM/repos/MM-WAM-Research
```

它是私有 GitHub 仓库。不得把 token 写入命令、日志或配置。优先使用 GitHub CLI/SSH 的
交互授权；若授权回调不可用，可以在已登录浏览器下载 ZIP 后上传，但该副本没有 `.git`
历史，只适合只读审计。

### 3.3 已定位的主工程与仍需核实的接口

同事没有提供工程代码，因此主工程已收敛为官方 Mobi-pi `release`：

```text
/share/chensiyu/MobiWAM/repos/mobipi
commit 19b130b8ada3f7e029918449c12d433e9e629ca1
```

首次 vanilla 命令、checkpoint、task、layout/style、seed 和 horizon 已从官方源码及既有复现
记录固定。环境与单任务资产的具体配置见 `docs/MOBIPI_BOOTSTRAP_CN.md`。

仍不能从论文猜测的内容是：冻结 policy 的 12-D action 到双臂/底盘的精确语义、snapshot
完整覆盖范围、控制器内部状态，以及真机爱宝的同步调度和急停 API。这些字段要在 vanilla
复现后通过源码和实测填写到 `configs/interface_contract.yaml`，不能在此之前开始 WAM 训练。

## 4. 工程实现顺序

### Phase A：复现 vanilla policy

目标：在完全不改策略的情况下，用固定 seed 稳定复现已有视频。

每次 rollout 必须保存：

- 初始 snapshot 与 state hash；
- 所有相机帧、机器人状态、policy observation 和 action chunk；
- base/arm 实际执行命令与时间戳；
- task success、碰撞、接触丢失和不可逆失败；
- checkpoint hash、代码 commit、配置和随机种子；
- 视频。

验收：同一 snapshot、同一 seed、同一 checkpoint 可复现相同 policy action；保存的 action
可在不再次调用 policy 的情况下重放。

### Phase B：实现三类可执行候选

先实现 `E`，再实现 `A(alpha)`，最后实现 `D(p)`：

1. `E` 是当前 action chunk 的原样执行，作为所有比较的锚点。
2. `A(0)` 必须逐数值等于 `E`；随后只测试小幅、无接触的底盘速度。
3. `A(alpha)` 中底盘位移引起的末端漂移必须由机械臂补偿，并记录位置/姿态误差。
4. `D(p)` 必须完成 save、navigate、settle、清历史、重新观测、重新询问 policy、execute。
5. 任意无效、碰撞或超约束候选保留在原始数据中，但不可进入选择集合。

### Phase C0：100 组 paired E/D/A 数据

先从 100 个相同 precontact source states 分叉，每个 state 执行一个 canonical `E`、`D`、
`A`，得到恰好 300 条 rollout。`E` 与 `A` 共享同一个 nominal policy chunk；`D` 必须在
post-dock observation 上 re-query。`X` 不执行，由没有安全成功 route 的 source state 派生。

这 300 条用于验证 collector、schema 和自动标签，不足以单独支持论文级 route-oracle claim。
详细协议见 `docs/PAIRED_EDA_COLLECTION_CN.md`。

### Phase C1：Route Oracle（WAM 之前的硬门）

从 25 个相同来源状态分叉，每个状态评估：

```text
1 × E + 3 × D(p) + 3 × A(alpha)，每个候选 3 个 seed
```

共 `25 × 7 × 3 = 525` 个短 rollout。比较 route oracle 与最佳固定路线。

继续条件：

- 最优路线不是几乎恒定同一种；
- route oracle 比最佳固定路线至少高 10 个成功率百分点；
- 失败类型中至少存在 WAM 可能从视觉后果识别的因素，如可见性、遮挡、碰撞风险或
  操作阶段进展。

不满足时先修改候选生成和任务分布，不训练 WAM。

### Phase D：先做便宜 baseline，再接 WAM

固定相同候选、相同 observation 和相同计算预算，依次实现：

1. 手工/几何规则；
2. value-only 分类器；
3. DreamTrajectory 风格的 trajectory-only evaluator；
4. 普通未来视频/特征 evaluator；
5. WAM；
6. 只有在 2D 视频无法表达关键几何失败时，才增加 4D-WAM 表征。

WAM 输入至少包含当前视觉、语言任务、机器人状态以及候选 base-arm action/trajectory；
输出用于预测 task progress、collision、visibility、reachability 和 uncertainty，最后排序
`E/D/A/X`。WAM 必须在同预算下优于 trajectory-only 和 value-only，才能成为主贡献。

### Phase E：仿真到真机

真机前先固定：速度/加速度限制、同步误差上限、过期 waypoint 处理、急停 API、相机标定、
底盘里程计和双臂末端误差。先做无接触动作，再做单臂接触，最后做双臂和长时任务。

## 5. GPU 使用规则

只在真正运行 GPU 任务时使用：

```bash
cd /share/chensiyu/MobiWAM
bash scripts/gpu_lease_status.sh
bash scripts/gpu_with_lease.sh -- <项目 GPU 命令>
```

包装器只精确匹配：

```text
/share/chensiyu/CoTTA/streamingqa/scripts/pi0.5_test.py
```

2026-08-26 晚间观察到的占卡程序是一个多进程树：父进程 `2402369`、资源跟踪器
`2402460` 和四个 worker `2402461--2402464`。多进程租约包装器会记录整棵树，先向父
进程和仍存活的后代发送 `SIGTERM`，确认所有原 PID 都退出后才运行项目命令；命令退出
后按原解释器、参数、工作目录和非敏感运行环境白名单重启父进程，并核对恢复后的后代
数量。它禁止使用 `SIGKILL`。CPU 配置和测试不需要停止真实占卡程序。

多进程回归测试可运行：

```bash
bash scripts/test_gpu_with_lease.sh
```

该测试只操作“一个父进程、一个资源跟踪器、四个 spawn worker”的假进程树，不接触
GPU 或真实占卡程序。2026-08-26 已通过该测试，并确认测试前后真实占卡树完全不变。
随后新增了“只保存非敏感运行环境白名单”的安全修订；该最终版本需要先同步到服务器并
重跑同一测试，才能用于真实占卡程序。

## 6. 当前状态（2026-08-27）

- 服务器为 4 张 A800 80GB，驱动 535.261.03；
- 项目骨架已部署到 `/share/chensiyu/MobiWAM`；
- GPU 多进程租约的树控制版本通过假进程停止、恢复和无残留检查；环境白名单安全修订待
  同步后复测，当前仍禁止用于真实占卡程序；
- Route Oracle CPU 测试 2/2 通过；
- 占卡程序仍运行；当前父 PID 为 `2402369`，另有资源跟踪器和四个 worker；
- Conda 环境已创建，项目可编辑安装且测试 2/2 通过；
- 私有研究仓库已部署到 `repos/MM-WAM-Research`（浏览器 ZIP 快照，无 `.git` 历史）；
- 项目目录已从旧名称改为 `/share/chensiyu/MobiWAM`，服务器文本引用已同步；
- 官方 Mobi-pi `release` 及六个 submodule 已完整部署并固定 commit；
- 服务器实测主环境为 Python 3.10.20、RoboCasa 0.2.0、robosuite 1.5.0、MuJoCo 3.2.6；
- RoboCasa365 已确定为 v1.0.1 / Python 3.11 / robosuite master 的隔离 transfer stack，
  不能覆盖主环境，详见 `docs/ROBOCASA_PLATFORM_DECISION_CN.md`；
- 主方法名已统一为 `E/D/A/X`；首批采集定义为 100 个 paired source states、300 条 rollout；
- 同事没有额外工程代码，Mobi-pi 已确定为仿真与 frozen policy 主仓库；
- vanilla evaluator 的精确命令已恢复；checkpoint-only loader 已确认首次评估不需要训练 HDF5；
- 独立 `envs/mobipi`、单任务资产与 CLIP/RoboCasa 配置脚本已在本地准备，等待同步执行；
- Mobi-pi CloseSingleDoor checkpoint 已就绪，HDF5 可后台续传但不阻塞首次评估；仍需由
  `--stage ready` 确认 CLIP 与 RoboCasa 资产后才能运行 vanilla GPU smoke。
