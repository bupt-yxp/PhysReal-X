---
title: 能力与方法
description: PhysReal-X 的方法契约、开源实现选型和接入边界
---

# 能力与方法

PhysReal-X 固定资源和业务语义，但不把不同算法伪装成同一种实现。一个方法只实现一个
`MethodCapability`；跨阶段流程由配方组合，算法代码整体收录在 `algorithms/<method>/`，
并由项目内适配器执行。

```text
能力契约（做什么） → 方法描述符（用什么实现） → 配方（按什么顺序执行）
```

## 方法契约

`MethodDescriptor` 固定能力、名称、版本、API 版本、源码地址与 revision、SPDX 许可证、
输入输出契约和运行要求；`MethodSpec` 再绑定权重制品与冻结配置。

```python
from physrealx import MethodCapability

installed = px.methods.installations(MethodCapability.ARTICULATION_RECOVER)
ready = tuple(item for item in installed if item.ready)
```

算法来源必须固定到明确 revision，带权重的方法必须先把权重写入制品存储。平台不在业务
契约中计算文件哈希；代码 revision、依赖版本、资源版本和制品标识分别承担来源、环境、业务
版本与存储定位职责。

## 项目内方法运行时

所有开源算法统一在当前 PhysReal-X 环境中按可选依赖安装：

```bash
python -m pip install -e '.[artgs]'
```

源码保留原始许可证并整体收录到 `algorithms/`；`src/physrealx/integrations/` 只负责平台契约转换。
当前运行模式为 `in_process`，所有方法共享项目的已解析依赖环境。

当前发行版只包含以下可执行实现，文档后续出现的其他项目均是**候选适配器**，不会因为名称
出现在表格或配置中就自动变为可运行能力：

| 能力 | 当前实现 | 边界 |
| --- | --- | --- |
| `observation.import` | `artmodeling` | 内置数据包导入器 |
| `geometry.reconstruct` | `artgs` | 源码已内嵌；需安装 `.[artgs]` 并满足 CUDA 探针 |
| `simulation.run` | `reference_drawer` | 仅用于平台契约回归，不是物理仿真器 |
| `policy.inference` | `scripted_drawer` | 仅用于契约回归 |
| `policy.train` / `policy.inference` | `torch_bc` | 安装 `.[train]` 后提供的数值基线 |

`px.methods.installations()` 是唯一运行时事实来源。候选方法只有在源码进入 `algorithms/`、依赖进入
根项目 extra、适配器注册并通过探针后，才能写入可执行配置。

```python
from physrealx import PhysRealX

px = PhysRealX.open(".physrealx")
installation = px.methods.inspect(MethodCapability.GEOMETRY_RECONSTRUCT, "artgs")
assert installation.ready
```

平台在执行前验证方法身份、输入契约和权重制品，执行后验证输出名称、语义契约、媒体类型、
声明大小和质量指标，再原子发布 `OperationManifest`。方法缺失、依赖不可用、输出越界或契约
不符均失败，不发布伪结果。
公共能力的输出不由调用者传入；服务层固定版本化 Schema、输出名和媒体类型，
`MethodDescriptor.output_contract` 与运行请求不一致时会在执行前被拒绝。

```python
installed = px.methods.installations(MethodCapability.GEOMETRY_RECONSTRUCT)
assert installed[0].ready
```

可选方法没有 SDK 默认值：调用必须显式传入已注册的 `method`，避免把候选名称误解为可执行能力。
操作输入同样不允许使用可变路径：必须先将观测、轨迹、标定、权重和上游产物发布为版本化
资源，再传入其 `ArtifactRef`。

## 1. 已采集数据到物理资产

目标编排顺序：

```text
预检与导入 → 坐标/单位规范化 → 相机恢复 → 米制对齐 → 可选深度增强 → 几何重建
→ 部件与关节恢复 → 碰撞构建 → 单关节物性辨识 → 完整资产组装 → OpenUSD 封装 → 质量验收
```

| 能力 | 候选实现与用途 |
| --- | --- |
| `observation.import` | 内置 `artmodeling` 校验 ArtModeling 包并发布不可变观测集。 |
| `camera.recover` | [COLMAP](https://github.com/colmap/colmap) 恢复多视角相机位姿与稀疏结构。 |
| `camera.align_metric` | RANSAC/Umeyama 适配器用独立米制对应解析 COLMAP Sim(3)。 |
| `depth.enhance` | [OMNI-DC](https://github.com/princeton-vl/OMNI-DC) 或 [CompletionFormer](https://github.com/youmi-zym/CompletionFormer)。 |
| `geometry.reconstruct` | 当前内嵌 `artgs`；Open3D TSDF 可作为后续适配器。 |
| `articulation.recover` | [Ditto](https://github.com/UT-Austin-RPL/Ditto) 或 [ScrewSplat](https://github.com/mbanani/screwsplat)。 |
| `collision.build` | [CoACD](https://github.com/SarahWeiii/CoACD) 生成近似凸碰撞代理。 |
| `physics.calibrate` | 自有 MuJoCo/SciPy 适配器可辨识单关节等效惯量、阻尼和摩擦。 |
| `asset.assemble` | 自有组装适配器需覆盖全部 link、body 与 joint；当前未提供。 |
| `asset.package` | [OpenUSD](https://github.com/PixarAnimationStudios/OpenUSD) 封装几何、关节、碰撞和物性。 |
| `asset.validate` | MuJoCo 留出轨迹物理探针。 |

`RDFC-GAN` 没有可直接复用的官方预训练权重，不作为默认深度方法。生成式资产不得覆盖实测
数字孪生；[Infinigen](https://github.com/princeton-vl/infinigen) 和 FreeArt3D 一类生成方法只能发布为
独立资产变体，并保留生成来源。

## 2. 场景与任务

目标编排顺序：

```text
资产筛选 → 场景生成/组合 → 约束求解 → 任务编译 → 可达性与碰撞验证
→ 轨迹生成 → 成功/失败/长尾变体
```

| 能力 | `method` | 开源实现与用途 |
| --- | --- | --- |
| `scene.generate` | `infinigen_indoor` | Infinigen 生成室内几何、材质和布局变体。 |
| `scene.generate` | `robocasa365` | [RoboCasa](https://github.com/robocasa/robocasa) 提供操作场景、任务和数据生成基线。 |
| `scene.compose` | `constraint_solver` | 候选确定性约束求解适配器，完成放置、间距、支撑和可达性检查。 |
| `task.compile` | `template` | 候选模板适配器，把结构化任务转为实体绑定、观测、动作与成功判据。 |
| `task.compile` | `bddl` | [BDDL](https://github.com/StanfordVL/bddl) 表达长时序活动条件。 |
| `task.plan` | `curobo` | [cuRobo](https://github.com/NVlabs/curobo) 生成 GPU 运动轨迹。 |
| `task.plan` | `ompl` | [OMPL](https://github.com/ompl/ompl) 提供便携运动规划基线。 |
| `trajectory.generate` | `robotwin2` | [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) 生成多任务、双臂示范。 |
| `task.validate` | `task_probe` | 候选验证适配器，在冻结初态、种子和超时下验证成功判据及轨迹可执行性。 |

自然语言模型只能生成结构化任务草案，不得直接执行任意代码；草案必须通过 Schema、实体解析、
约束求解、沙箱和仿真验证后才能发布。许可证限制为非商业用途的项目不进入默认方法集。
本节所有生成、编译、规划和验证方法都是适配器设计选项，核心仓库当前未实现这些方法。

## 3. 仿真与多模态数据

目标编排顺序：

```text
场景编译 → 确定性 reset → 策略/规划执行 → 物理与渲染采样
→ 模态时间对齐 → Episode 校验 → 数据集原子发布
```

| 能力 | `method` | 开源实现与用途 |
| --- | --- | --- |
| `simulation.run` | `mujoco` | MuJoCo 适配器候选；核心仓库未附带该实现，部署方必须实现场景/任务编译、真实时间步进与轨迹证据。 |
| `simulation.run` | `genesis` | [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) 用于 GPU 并行与多物理实验。 |
| `simulation.run` | `gazebo` | [Gazebo](https://github.com/gazebosim/gz-sim) 用于 ROS 2 系统仿真。 |
| `render.generate` | `blenderproc` | [BlenderProc](https://github.com/DLR-RM/BlenderProc) 适配器候选；核心仓库未附带该实现，必须遵守 GPL并严格重放输入物理轨迹。 |
| `event.synthesize` | `v2e` | [v2e](https://github.com/SensorsINI/v2e) 从强度视频生成事件流。 |
| `point_cloud.project` | `open3d` | Open3D 依据相机模型生成和检查点云。 |
| `randomization.apply` | `domain_randomizer` | 候选随机化适配器，按已记录随机实值改变材质、光照、相机、噪声和动力学。 |
| `episode.pack` | `lerobot_v3` | [LeRobot](https://github.com/huggingface/lerobot) v3 适配器候选；核心仓库未附带该实现，必须保留平台 Episode 的 Schema、时序和来源。 |

Isaac Sim/Isaac Lab 可作为受其许可证约束的可选方法，但不标为严格开源默认后端。每个后端只
声明自己真实支持的传感器、物理能力和确定性等级，不通过统一接口伪造缺失输出。
上表中 MuJoCo、Genesis、Gazebo、BlenderProc、v2e、Open3D 和 LeRobot 均是适配器设计选项，
不是本仓库已实现能力；可执行性仅以项目方法注册记录为准。

## 4. 数据、世界模型与策略训练

目标编排顺序：

```text
MCAP/仿真 Episode → LeRobot v3 规范化 → 校验与防泄漏切分 → 不可变快照
→ 受控训练 → 模型登记 → 冻结协议评测
```

| 能力 | `method` | 开源实现与用途 |
| --- | --- | --- |
| `dataset.ingest` | `lerobot_v3` | LeRobot v3 候选适配器；需以项目可选依赖安装并把转换、质检和重读验证结果发布为证据。 |
| `dataset.ingest` | `rlds` | [RLDS](https://github.com/google-research/rlds) 只用于外部 Episode 交换。 |
| `dataset.snapshot` | `dvc` | [DVC](https://github.com/treeverse/dvc) 对大型数据制品做数据版本与远端存储。 |
| `model.register` | `mlflow` | [MLflow](https://github.com/mlflow/mlflow) 登记数据、代码、环境、配置、指标和权重。 |
| `world_model.train` | `dino_wm` | [DINO-WM](https://github.com/gaoyuezhou/dino_wm) 学习动作条件特征动力学与 CEM 规划。 |
| `world_model.train` | `irasim` | [IRASim](https://github.com/bytedance/IRASim) 根据首帧与动作生成未来机器人视频。 |
| `world_model.train` | `dreamerv3` | [DreamerV3](https://github.com/danijar/dreamerv3) 执行潜在动力学、想象 rollout 与 actor-critic。 |
| `world_model.train` | `tdmpc2` | [TD-MPC2](https://github.com/nicklashansen/tdmpc2) 提供连续控制的模型预测基线。 |
| `policy.train` | `lerobot_act` | LeRobot ACT 候选适配器；需以项目可选依赖安装、固定数据/代码/环境并通过固定基准。 |
| `policy.train` | `diffusion_policy` | [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) 建模多峰连续动作。 |
| `policy.train` | `vqbet` | [VQ-BeT](https://github.com/jayLEE0301/vq_bet_official) 学习离散动作意图与连续控制。 |
| `policy.train` | `robomimic_bc` | [robomimic](https://github.com/ARISE-Initiative/robomimic) 提供 BC/离线学习复现基线。 |
| `policy.train` | `rsl_rl_ppo` | [RSL-RL](https://github.com/leggedrobotics/rsl_rl) 提供 GPU PPO 与蒸馏。 |

PPT 中 CAPE 的功能由 DINO-WM、IRASim、DreamerV3 和 TD-MPC2 覆盖；“VQ-VAE +
Diffusion/FM”拆成 VQ-BeT、Diffusion Policy 和流匹配方法独立比较，不能虚构成一个已有项目。
OpenVLA/openpi 的代码与模型权重许可证必须分别审核，默认方法不自动下载这些权重。
本节列出的数据转换、世界模型和机器人策略均未作为核心仓库的已实现方法；内置 `torch_bc` 只是数值与契约回归基线。

## 5. 评测、部署与失败反馈

目标编排顺序：

```text
离线回放 → 固定仿真基准 → shadow → 限速/限力 canary → 真机运行
→ MCAP 留证 → 故障归因 → 定向数据集 → 再训练与全量回归
```

| 能力 | `method` | 开源实现与用途 |
| --- | --- | --- |
| `evaluation.run` | `libero` | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 评测多任务与终身学习策略。 |
| `evaluation.run` | `robotwin2` | RoboTwin 2.0 评测多本体与双臂操作。 |
| `evaluation.run` | `simpler` | [SimplerEnv](https://github.com/simpler-env/SimplerEnv) 检查仿真与真机表现相关性。 |
| `deployment.stage` | `ros2_control` | [ros2_control](https://github.com/ros-controls/ros2_control) 与 [MoveIt 2](https://github.com/moveit/moveit2) 接入真机。 |
| `telemetry.record` | `rosbag2_mcap` | rosbag2 + MCAP 记录观测、动作、TF、诊断、接触和任务判据。 |
| `failure.analyze` | `rules_v1` | 平台规则以时间线、跟踪误差、接触/力和诊断生成可核验故障证据。 |
| `failure.analyze` | `robofac` | [RoboFAC](https://github.com/MINT-SJTU/RoboFAC) 作为二级视频 critic，输出须经规则或人工确认。 |
| `feedback.build_dataset` | `hard_negative` | 从确认失败/OOD 切片发布新的不可变训练集版本。 |

学习策略只输出期望动作或轨迹；急停、工作空间、关节/速度/力矩/力限制和 watchdog 由确定性
安全监督器强制执行。故障无法确认时必须保留 `unknown`，不得强行归因。
上表为部署设计选项；核心仓库未附带 LIBERO、ROS 2、MCAP、RoboFAC 或反馈数据生成实现。

## 发布门禁

1. 方法必须登记可核验源码 revision、许可证、输入输出 Schema 和完整运行要求。
2. 算法通过可选依赖安装到项目环境；冲突依赖必须在安装阶段明确失败，不创建隐式环境。
3. 数据、模型和报告均绑定输入资源、方法描述符、配置、种子、代码与环境版本。
4. 资产通过结构、碰撞、动力学和回放门禁；数据通过时间同步、Schema、制品引用与防泄漏门禁。
5. 策略报告逐任务成功率、置信区间、延迟和安全事件；真机发布必须通过独立回归集。
6. ArtGen、PD²GS、CAPE 等无法核验官方可复用代码、权重和许可证的工作不注册为可执行方法。
