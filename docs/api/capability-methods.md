---
title: 能力与方法
description: PhysReal-X 的方法契约、开源实现选型和接入边界
---

# 能力与方法

PhysReal-X 固定资源和业务语义，但不把不同算法伪装成同一种实现。一个方法只实现一个
`MethodCapability`；跨阶段流程由配方组合，方法由独立 worker 或适配器执行。

```text
能力契约（做什么） → 方法描述符（用什么实现） → 配方（按什么顺序执行）
```

## 方法契约

`MethodDescriptor` 固定能力、名称、版本、API 版本、源码地址与不可变 revision、SPDX
许可证、输入输出契约和运行环境；`MethodSpec` 再绑定权重摘要与冻结配置。

```python
from physrealx import MethodCapability

available = px.methods.list(MethodCapability.ARTICULATION_RECOVER)
ditto = px.methods.bind(
    MethodCapability.ARTICULATION_RECOVER,
    "ditto",
    config={"joint_types": ["revolute", "prismatic"]},
)
```

容器方法必须登记镜像 SHA-256；带权重的方法必须先将权重写入制品存储。`main`、`latest`
或本地可变路径不能作为来源版本。配方中记录完整描述符，运行前再与当前已安装描述符精确比对。

## 独立 worker 运行时

Ditto、CoACD、MuJoCo、BlenderProc、LeRobot 等不进入核心依赖。每个方法安装自己的算法环境；
本机进程通过工作区 `workers.d` 部署清单绑定，发布扩展也可通过 `physrealx.workers`
入口登记。只有描述符而无可执行环境的方法不会被标记为就绪。

运行模式只有三种：

| 模式 | 用途 | 隔离要求 |
| --- | --- | --- |
| `in_process` | 平台内置、轻量且可信的参考实现 | 不得引入 GPU/仿真框架依赖 |
| `process` | 独立 Conda/venv 中的开发与受控部署 | 使用绝对可执行文件、无 shell、超时和有限日志 |
| `container` | 生产算法 worker | OCI 镜像必须使用 digest 固定，默认无网络、只读根目录、移除 capabilities |

进程 worker 的扩展注册入口：

```python
from pathlib import Path

from physrealx.external_workers import ExternalMethodWorker, ProcessLaunch


def register(registry):
    registry.register_worker(
        ExternalMethodWorker(
            coacd_descriptor,
            ProcessLaunch(
                command=("/opt/physrealx-coacd/bin/physrealx-coacd",),
                environment_manifest=Path("/opt/physrealx-coacd/environment.lock.json"),
            ),
        )
    )
```

扩展包在自己的 `pyproject.toml` 中声明：

```toml
[project.entry-points."physrealx.workers"]
coacd = "physrealx_coacd:register"
```

worker 可使用 `run_worker(handler)` 实现统一命令入口。handler 读取已校验的请求与只读输入，
只把输出写入指定目录：

```python
from physrealx.external_workers import WorkerProducedFile, WorkerResponse, run_worker


def handler(request, output_dir):
    source = output_dir.parent / request.inputs[0].path
    output = run_coacd(source, output_dir / "collision.glb")
    return WorkerResponse(
        outputs=(
            WorkerProducedFile(
                name="collision_geometry",
                contract="CollisionGeometry",
                path=output.name,
                media_type="model/gltf-binary",
            ),
        ),
        metrics={"part_count": 12.0},
    )


raise SystemExit(run_worker(handler))
```

`process` 描述符的 `environment_digest` 必须等于 `environment_manifest` 文件的
SHA-256；该文件应固定 Python、CUDA、系统库、算法包和权重版本。生产部署优先使用
`container`，因为镜像 digest 能够完整固定文件系统。

平台在执行前验证 worker 就绪状态、方法身份、输入与权重摘要；执行后验证协议版本、输出名称、
语义契约、相对路径、文件类型、大小和摘要，再原子发布 `OperationManifest`。同一精确请求重复调用
返回原证据；超时、非零退出、缺失镜像、路径越界或契约不符均失败，不发布伪结果。

```python
installed = px.methods.installations(MethodCapability.COLLISION_BUILD)
assert installed[0].ready
```

外部方法没有 SDK 默认值：调用必须显式传入已安装的 `method`，避免把推荐名称误解为可执行能力。

## 1. 已采集数据到物理资产

默认流程：

```text
预检与导入 → 坐标/单位规范化 → 相机恢复 → 可选深度增强 → 几何重建
→ 部件与关节恢复 → 碰撞构建 → 物性标定 → OpenUSD 封装 → 质量验收
```

| 能力 | `method` | 开源实现与用途 |
| --- | --- | --- |
| `observation.import` | `artmodeling` | 平台内置导入器；校验 ArtModeling 包并发布不可变观测集。 |
| `camera.recover` | `colmap` | [COLMAP](https://github.com/colmap/colmap) 恢复多视角相机位姿与稀疏结构。 |
| `depth.enhance` | `omni_dc` | [OMNI-DC](https://github.com/princeton-vl/OMNI-DC) 使用已发布代码和权重补全跨域深度。 |
| `depth.enhance` | `completionformer` | [CompletionFormer](https://github.com/youmi-zym/CompletionFormer) 处理匹配 NYUv2/KITTI 数据域的 RGB-D。 |
| `geometry.reconstruct` | `open3d_fusion` | [Open3D](https://github.com/isl-org/Open3D) 对已标定 RGB-D 执行 TSDF/点云融合。 |
| `articulation.recover` | `ditto` | [Ditto](https://github.com/UT-Austin-RPL/Ditto) 从交互前后点云恢复部件和关节。 |
| `articulation.recover` | `screwsplat` | [ScrewSplat](https://github.com/mbanani/screwsplat) 从多状态视觉恢复可动结构；作为实验方法隔离。 |
| `collision.build` | `coacd` | [CoACD](https://github.com/SarahWeiii/CoACD) 生成近似凸碰撞代理。 |
| `physics.calibrate` | `mujoco_sysid` | [MuJoCo](https://github.com/google-deepmind/mujoco) 与优化器反演质量、摩擦、阻尼和接触参数。 |
| `asset.package` | `openusd` | [OpenUSD](https://github.com/PixarAnimationStudios/OpenUSD) 封装几何、关节、碰撞和物性。 |
| `asset.validate` | `physics_probe` | 平台验证器配合 MuJoCo 回放检查结构、穿模、限位、接触和动力学误差。 |

`RDFC-GAN` 没有可直接复用的官方预训练权重，不作为默认深度方法。生成式资产不得覆盖实测
数字孪生；[Infinigen](https://github.com/princeton-vl/infinigen) 和 FreeArt3D 一类生成方法只能发布为
独立资产变体，并保留生成来源。

## 2. 场景与任务

默认流程：

```text
资产筛选 → 场景生成/组合 → 约束求解 → 任务编译 → 可达性与碰撞验证
→ 轨迹生成 → 成功/失败/长尾变体
```

| 能力 | `method` | 开源实现与用途 |
| --- | --- | --- |
| `scene.generate` | `infinigen_indoor` | Infinigen 生成室内几何、材质和布局变体。 |
| `scene.generate` | `robocasa365` | [RoboCasa](https://github.com/robocasa/robocasa) 提供操作场景、任务和数据生成基线。 |
| `scene.compose` | `constraint_solver` | 平台确定性约束求解器完成放置、间距、支撑和可达性检查。 |
| `task.compile` | `template` | 平台模板把结构化任务转为实体绑定、观测、动作与成功判据。 |
| `task.compile` | `bddl` | [BDDL](https://github.com/StanfordVL/bddl) 表达长时序活动条件。 |
| `task.plan` | `curobo` | [cuRobo](https://github.com/NVlabs/curobo) 生成 GPU 运动轨迹。 |
| `task.plan` | `ompl` | [OMPL](https://github.com/ompl/ompl) 提供便携运动规划基线。 |
| `trajectory.generate` | `robotwin2` | [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) 生成多任务、双臂示范。 |
| `task.validate` | `task_probe` | 平台在冻结初态、种子和超时下验证成功判据及轨迹可执行性。 |

自然语言模型只能生成结构化任务草案，不得直接执行任意代码；草案必须通过 Schema、实体解析、
约束求解、沙箱和仿真验证后才能发布。许可证限制为非商业用途的项目不进入默认方法集。

## 3. 仿真与多模态数据

默认流程：

```text
场景编译 → 确定性 reset → 策略/规划执行 → 物理与渲染采样
→ 模态时间对齐 → Episode 校验 → 数据集原子发布
```

| 能力 | `method` | 开源实现与用途 |
| --- | --- | --- |
| `simulation.run` | `mujoco` | MuJoCo 作为轻量、可复现的刚体与接触默认后端。 |
| `simulation.run` | `genesis` | [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) 用于 GPU 并行与多物理实验。 |
| `simulation.run` | `gazebo` | [Gazebo](https://github.com/gazebosim/gz-sim) 用于 ROS 2 系统仿真。 |
| `render.generate` | `blenderproc` | [BlenderProc](https://github.com/DLR-RM/BlenderProc) 离线生成 RGB、Depth、Mask；GPL worker 必须隔离。 |
| `event.synthesize` | `v2e` | [v2e](https://github.com/SensorsINI/v2e) 从强度视频生成事件流。 |
| `point_cloud.project` | `open3d` | Open3D 依据相机模型生成和检查点云。 |
| `randomization.apply` | `domain_randomizer` | 平台按已记录随机实值改变材质、光照、相机、噪声和动力学。 |
| `episode.pack` | `lerobot_v2_1` | [LeRobot](https://github.com/huggingface/lerobot) v2.1 组织 Parquet、时序与元数据。 |

Isaac Sim/Isaac Lab 可作为受其许可证约束的可选 worker，但不标为严格开源默认后端。每个后端只
声明自己真实支持的传感器、物理能力和确定性等级，不通过统一接口伪造缺失输出。

## 4. 数据、世界模型与策略训练

默认流程：

```text
MCAP/仿真 Episode → LeRobot v2.1 规范化 → 校验与防泄漏切分 → 不可变快照
→ 隔离训练 → 模型登记 → 冻结协议评测
```

| 能力 | `method` | 开源实现与用途 |
| --- | --- | --- |
| `dataset.ingest` | `lerobot_v2_1` | 作为 NumPy 1.26 环境的训练格式，统一低维时序、视频和元数据。 |
| `dataset.ingest` | `rlds` | [RLDS](https://github.com/google-research/rlds) 只用于外部 Episode 交换。 |
| `dataset.snapshot` | `dvc` | [DVC](https://github.com/treeverse/dvc) 对大型数据制品做内容寻址快照。 |
| `model.register` | `mlflow` | [MLflow](https://github.com/mlflow/mlflow) 登记数据、代码、环境、配置、指标和权重。 |
| `world_model.train` | `dino_wm` | [DINO-WM](https://github.com/gaoyuezhou/dino_wm) 学习动作条件特征动力学与 CEM 规划。 |
| `world_model.train` | `irasim` | [IRASim](https://github.com/bytedance/IRASim) 根据首帧与动作生成未来机器人视频。 |
| `world_model.train` | `dreamerv3` | [DreamerV3](https://github.com/danijar/dreamerv3) 执行潜在动力学、想象 rollout 与 actor-critic。 |
| `world_model.train` | `tdmpc2` | [TD-MPC2](https://github.com/nicklashansen/tdmpc2) 提供连续控制的模型预测基线。 |
| `policy.train` | `lerobot_act` | LeRobot ACT 作为默认示范学习基线。 |
| `policy.train` | `diffusion_policy` | [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) 建模多峰连续动作。 |
| `policy.train` | `vqbet` | [VQ-BeT](https://github.com/jayLEE0301/vq_bet_official) 学习离散动作意图与连续控制。 |
| `policy.train` | `robomimic_bc` | [robomimic](https://github.com/ARISE-Initiative/robomimic) 提供 BC/离线学习复现基线。 |
| `policy.train` | `rsl_rl_ppo` | [RSL-RL](https://github.com/leggedrobotics/rsl_rl) 提供 GPU PPO 与蒸馏。 |

PPT 中 CAPE 的功能由 DINO-WM、IRASim、DreamerV3 和 TD-MPC2 覆盖；“VQ-VAE +
Diffusion/FM”拆成 VQ-BeT、Diffusion Policy 和流匹配方法独立比较，不能虚构成一个已有项目。
OpenVLA/openpi 的代码与模型权重许可证必须分别审核，默认方法不自动下载这些权重。

## 5. 评测、部署与失败反馈

默认流程：

```text
离线回放 → 固定仿真基准 → shadow → 限速/限力 canary → 真机运行
→ MCAP 留证 → 故障归因 → 定向数据集 → 再训练与全量回归
```

| 能力 | `method` | 开源实现与用途 |
| --- | --- | --- |
| `evaluation.run` | `libero` | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 评测多任务与终身学习策略。 |
| `evaluation.run` | `robotwin2` | RoboTwin 2.0 评测多本体与双臂操作。 |
| `evaluation.run` | `simpler` | [SimplerEnv](https://github.com/simpler-env/SimplerEnv) 检查仿真与真机表现相关性。 |
| `deployment.stage` | `ros2` | [ros2_control](https://github.com/ros-controls/ros2_control) 与 [MoveIt 2](https://github.com/moveit/moveit2) 接入真机。 |
| `telemetry.record` | `rosbag2_mcap` | rosbag2 + MCAP 记录观测、动作、TF、诊断、接触和任务判据。 |
| `failure.analyze` | `rules` | 平台规则以时间线、跟踪误差、接触/力和诊断生成可核验故障证据。 |
| `failure.analyze` | `robofac` | [RoboFAC](https://github.com/MINT-SJTU/RoboFAC) 作为二级视频 critic，输出须经规则或人工确认。 |
| `feedback.build_dataset` | `hard_negative` | 从确认失败/OOD 切片发布新的不可变训练集版本。 |

学习策略只输出期望动作或轨迹；急停、工作空间、关节/速度/力矩/力限制和 watchdog 由确定性
安全监督器强制执行。故障无法确认时必须保留 `unknown`，不得强行归因。

## 发布门禁

1. 方法必须登记可核验源码、不可变版本、许可证、输入输出 Schema 和完整运行要求。
2. 算法环境与 Python 3.8–3.12 核心环境隔离；CUDA、系统库和模型权重不得进入核心依赖。
3. 数据、模型和报告均绑定输入摘要、方法描述符、配置、种子、代码与环境版本。
4. 资产通过结构、碰撞、动力学和回放门禁；数据通过时间同步、Schema、摘要与防泄漏门禁。
5. 策略报告逐任务成功率、置信区间、延迟和安全事件；真机发布必须通过独立回归集。
6. ArtGen、PD²GS、CAPE 等无法核验官方可复用代码、权重和许可证的工作不注册为可执行方法。
