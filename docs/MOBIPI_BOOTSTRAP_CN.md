# Mobi-pi 从零部署手册

## 结论先说

当前工程的第一层不是 DreamTrajectory，也不是 WAM。第一层是官方 Mobi-pi `release`
里的 RoboCasa 仿真和冻结 manipulation policy；DreamTrajectory 以后只作为候选轨迹打分
baseline，WAM/4D-WAM 在 Route Oracle 证明“不同状态应选不同路线”后再接入。

### 2026-08-27 checkpoint-only eval 修正

实际 checkpoint 已确认包含 `env_metadata`、`shape_metadata` 和
`action_normalization_stats`（`scale/offset` 均为 `(1,12)`）。因此 vanilla smoke、A(0) 和首轮
paired collection 不再被 9.60 GB source HDF5 阻塞：`mobiwam.mobipi_checkpoint` 直接读取训练时
随 checkpoint 保存的同一组 metadata/stats，再走原 `RolloutPolicy` 反归一化逻辑。

完整 HDF5 仍可后台续传，用于以后重训和“checkpoint stats == dataset stats”交叉审计，但不再是
第一次 rollout 的前置条件。任何 loader 若缺少上述 checkpoint fields 必须失败，不能退回无
归一化 action。

服务器正式目录已经去掉 `ICRA27`：

```text
/share/chensiyu/MobiWAM
```

官方源码已部署到：

```text
/share/chensiyu/MobiWAM/repos/mobipi
```

根 commit 固定为 `19b130b8ada3f7e029918449c12d433e9e629ca1`，六个 submodule 的
commit 由 `scripts/bootstrap_mobipi_env.sh` 逐个核验。不要在 `release` 上执行 `pull` 或
重新初始化 submodule。

## 为什么不直接运行官方 install.sh

官方脚本可以作为依赖清单，但不适合在共享服务器整段执行：

- 它会安装 pre-commit hook 并改变仓库运行状态；
- 它在创建环境时立刻下载约 5 GB RoboCasa 资产；
- bundled robomimic 声明 `torch==2.0.1`，根脚本最后又要求 `torch==2.2.0`；
- 它把 vanilla、LeLaN、3DGS/nerfstudio 一次装完，出错后难以定位；
- 多个依赖未锁版本，也没有记录 module source 和完整 freeze。

本项目保留已验证的 Python `3.10.20`、PyTorch `2.2.0+cu121`、NumPy `1.23.3`，
把环境、路径、资产和 GPU rollout 分成四个可独立验收的阶段。首个 vanilla rollout 不需要
nerfstudio；3DGS/Mobi-pi 完整方法以后作为单独 baseline 环境扩展。

## 阶段 1：安装独立环境

该阶段只联网、下载 Python 包和做 CPU import，不使用显卡，不停止占卡程序。

```bash
cd /share/chensiyu/MobiWAM
bash scripts/bootstrap_mobipi_env.sh
```

成功标准：

- Python 位于 `/share/chensiyu/MobiWAM/envs/mobipi/bin/python`；
- `audit/mobipi_environment_preflight.json` 的 `status` 为 `pass`；
- `audit/mobipi_pip_freeze.txt` 和 `audit/mobipi_source_commits.txt` 已生成；
- `mobipi`、RoboCasa、robomimic 等模块只来自本环境或固定源码树，不来自 `~/.local`。

`pip check` 预期只保留 bundled robomimic 的三条 metadata 冲突：它声明旧版 NumPy、Torch
和 torchvision，但官方 Mobi-pi 根脚本及已完成的历史复现使用新组合。出现第四条冲突即
视为失败。

## 阶段 2：下载第一个任务所需资产

### 2.1 Checkpoint 和训练 HDF5

源码检查确认，未修改的官方 `eval_baseline.py` 会打开训练 HDF5；但实际 checkpoint 已带相同的
环境元数据、shape 和 normalization stats。当前 `ready` 门和 checkpoint-only vanilla 只要求
`CloseSingleDoor / seed 1` checkpoint，不下载其余十四个 checkpoint 或其余四任务数据。

```bash
cd /share/chensiyu/MobiWAM
envs/mobipi/bin/python scripts/download_mobipi_close_single_door.py
```

完整 HDF5 可继续后台下载，完成后用于 upstream parity / retraining，并进入 `full` 门：

```bash
envs/mobipi/bin/python scripts/download_mobipi_close_single_door.py --with-dataset
```

脚本支持断点续传并保留校验失败文件，不静默覆盖。验收锚点：

| 资产 | 校验 |
|---|---|
| checkpoint ZIP | SHA-256 `a1294905...3fe3a0d` |
| model file | `246511905` bytes；SHA-256 `6cafee55...be9852` |
| dataset ZIP | `5612812816` bytes；SHA-256 `b14a21a7...bc99c`（仅 `full`） |
| HDF5 | `9601187887` bytes；`300_demos` 必须正好 300 条 |

ZIP 保留在 `downloads/mobipi` 作为来源证据。checkpoint 的 `config.json` 会通过 JSON parser
把数据路径改为本项目目录；旧服务器记录的 config SHA 含旧绝对路径，不能作为新机器改写
后的跨机器 checksum。

### 2.2 RoboCasa 资产

此步骤使用 bundled RoboCasa 的官方 Box 下载器，并核验下载后 tracked source 没有变化。
它可能较慢，必须放在具名 tmux 中：

```bash
cd /share/chensiyu/MobiWAM
tmux new-session -d -s mobiwam-robocasa-assets \
  'cd /share/chensiyu/MobiWAM && bash scripts/download_robocasa_assets.sh'
tmux attach -t mobiwam-robocasa-assets
```

若 Box 连续无字节增长，停止本次下载并记录现场，再有限测速 HF 镜像。不得悄悄换源；历史
可用镜像与 bundled ZIP 大小不同，换用时必须登记为 protocol deviation。

### 2.3 CLIP 缓存

冻结 policy 需要 `openai/clip-vit-large-patch14`。先下载到项目自己的 Hugging Face cache，
再用已知权重大小和 SHA-256 校验：

```bash
cd /share/chensiyu/MobiWAM
HF_ENDPOINT=https://hf-mirror.com \
  envs/mobipi/bin/python scripts/cache_mobipi_clip.py
```

若镜像不可用，去掉 `HF_ENDPOINT` 使用官方端点。两者只能改变传输来源，最终权重必须是
`1710540580` bytes，SHA-256 `a2bf730a...026dcb`。

## 阶段 3：CPU 就绪门

```bash
cd /share/chensiyu/MobiWAM
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 \
  envs/mobipi/bin/python scripts/mobipi_preflight.py \
  --stage ready --output audit/mobipi_ready_preflight.json
```

只有 `status: pass` 才能进入 GPU。`ready` 会验证环境和 module source、源码 commit、private
macros、checkpoint、CLIP、RoboCasa 资产，以及官方 `eval_baseline.py` 顶层导入；只有
`--stage full` 才额外验证 9.60 GB HDF5。

## 阶段 4：单回合 vanilla GPU smoke

先同步并通过最终版 GPU lease 假进程测试。当前真实占卡程序是
`/share/chensiyu/CoTTA/streamingqa/scripts/pi0.5_test.py`，观察到的进程树为父进程
`2402369`、resource tracker 和四个 workers。PID 不是永久配置，wrapper 每次都按精确脚本
路径和 `/proc` start time 重新识别。

```bash
cd /share/chensiyu/MobiWAM
bash scripts/test_gpu_with_lease.sh
bash scripts/gpu_lease_status.sh
MOBIWAM_GPU_INDEX=0 bash scripts/gpu_with_lease.sh -- \
  bash scripts/run_mobipi_vanilla_once.sh
```

`run_mobipi_vanilla_once.sh` 禁止直接运行。lease wrapper 必须先完整记录并停止父进程、tracker
和四个 workers；rollout 退出或报错时都会恢复占卡程序并核对后代数量。首次任务固定为：

```text
task=CloseSingleDoor
policy=bc_xfmr / vanilla_policy
layout=1, style=1, checkpoint seed=1
episodes=1, horizon=500, base offset=0.00
```

成功标准不是“生成了一个视频”，而是 evaluator exit code 为 0、`Success_Rate` 可读、视频与
episode artifact 存在、manifest 中的 commit/checkpoint/GPU/lease audit 可回溯，并且占卡
程序恢复为一棵父进程加五个后代的健康进程树。

## 之后怎么接论文工程

完成 vanilla smoke 后按以下顺序开发，不能从 WAM 训练倒推接口：

1. 固定 observation/action/state/video 日志 schema，证明 action 可重放；
2. 在相同 snapshot 上实现 `E`，证明 `A(0) == E`；
3. 实现小幅在线辅助 `A(alpha)` 和 world-frame 末端补偿；
4. 实现 `D(p)` 的 navigate、settle、历史处理、重新询问 policy 全生命周期；
5. 从相同 source states 真实分叉 `E/D/A`；`X` 由无安全候选的结果派生，不执行 X rollout；
6. 先比较规则、value-only、DreamTrajectory-style trajectory-only；
7. 只有 route preference 非恒定且 oracle 明显优于固定路线时，接 WAM consequence evaluator；
8. 只有 2D 未来表征无法判断关键几何失败时，才增加 4D 表征。

这样 WAM 的作用是预测候选执行后果并选择干预方式，而不是给已有移动操作 pipeline 加一个
没有因果作用的模块。
