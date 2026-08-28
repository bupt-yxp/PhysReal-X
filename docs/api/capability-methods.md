---
title: 技术能力与方法 API
description: 从已采集观测、数字孪生到训练和真机反馈的能力接口与开源实现选型
---

# 技术能力与方法 API

本页把两份方案中的研究技术组织为可实现的平台能力；公共 API 只描述“做什么”，算法、仿真器和训练框架通过 `pipeline`、`backend`、`trainer` 或 `analyzer` 插件选择，不进入领域模型。

## 技术来源

| 功能模块 | 原版来源 | v3 来源 | 平台结果 |
| --- | --- | --- | --- |
| 真实资产接入与数字孪生 | P12–13、P28–34 | P5–6 | 已采集观测、数字孪生资产与多模态数据 |
| 物理可信资产库 | P13、P15、P35–42 | P7–8 | 可交互、可校准、可版本化资产 |
| 场景与任务生成 | P18 | P9 | 场景、任务、轨迹与随机化规格 |
| 物理仿真与多模态渲染 | P14–16、P19、P35–36 | P10–11 | Episode 与质量合格的数据集 |
| 数据管理、世界模型与策略训练 | P20–24、P44–50 | P12–13 | 数据集、世界模型、策略与检查点 |
| 评测与真机反馈闭环 | P22、P25–26 | P14 | 评测报告、真机记录与再生成计划 |
| 平台公共底座 | P17、P43 | P4、P16 | 统一契约、调度、存储、追踪与可视化 |

### 方案算法与开源状态

| 技术 | 接入结论 |
| --- | --- |
| [RDFC-GAN](https://github.com/midea-ai/RDFC-GAN) | 官方已开源训练/测试代码，但未提供可直接下载的预训练权重，不作默认运行方法。 |
| [DTF-Net](https://arxiv.org/abs/2308.02239) | 采用论文方法定义插件能力，未确认官方代码前使用 Self-DPDN 等开源基线。 |
| [OMNI-DC](https://github.com/princeton-vl/OMNI-DC) | 官方代码、自定义数据示例和预训练权重均可用，作为默认跨数据深度补全插件。 |
| [CompletionFormer](https://github.com/youmi-zym/CompletionFormer) | 官方提供 NYUv2/KITTI 权重，作为有明确数据域的基线。 |
| [VideoArtGS](https://videoartgs.github.io/) / [FreeArtGS](https://github.com/FreeArtGS/FreeArtGS) | 已发布代码，分别覆盖单目视频和自由运动铰接重建。 |
| [ArtGen](https://arxiv.org/abs/2512.12395) | 采用论文方法定义生成器能力，未确认官方代码前使用 Articulate Anything 与 Arti-PG。 |
| [CAPE](https://arxiv.org/abs/2606.07304) | 采用论文方法定义世界模型训练器，未确认官方代码前使用 DINOv2、PyTorch 与 OpenDW。 |
| VQ-VAE + Diffusion/FM | 这是策略技术组合，分别使用开源量化器、Diffusion Policy 和 Flow Matching 组件实现。 |

最终链路：

```text
已采集观测 → 数字孪生 → 物理资产库 → 场景任务 → 仿真数据 → 世界模型/策略训练 → 真机评测 → 定向再生成
```

## 1. 真实资产接入与数字孪生

| API | 用途 | 推荐开源实现 |
| --- | --- | --- |
| `assets.inspect_source(path, importer=...)` | 只读检查已采集数据包的结构、帧对齐、标定缺失和安全性。 | ArtModeling importer、NPY 头解析 |
| `assets.ingest(path, importer=...)` | 规范化模态语义并将原始文件发布为不可变观测集。 | Pydantic、SHA-256 内容寻址存储 |
| `assets.build(DigitalTwinBuildSpec(...))` | 按逐阶段方法配方生成几何、部件、关节、物性和 USD 资产。 | COLMAP/VGGT、OMNI-DC、ArtGS/VideoArtGS、CoACD、OpenUSD |
| `assets.validate(asset, profile="digital_twin")` | 在发布前检查制品、link/joint、碰撞表示、物性来源和限制。 | PhysReal-X validator、MuJoCo |

实现约定：平台不控制采集设备；深度、相机、铰接、物性和封装方法分阶段声明。ArtGS 类方法只实现 `articulated_reconstruction`，不冒充整体 pipeline。

## 2. 物理可信资产库

| API | 用途 | 推荐开源实现 |
| --- | --- | --- |
| `assets.analyze_articulation(asset, analyzer=...)` | 识别部件层级、关节类型、运动轴、原点和运动范围。 | Articulate Anything、PartNet-Mobility、Open3D |
| `assets.build(spec)` 的铰接重建阶段 | 从多状态观测恢复部件解耦、连续形变和运动学关系。 | ArtGS、VideoArtGS、FreeArtGS、gsplat |
| `assets.generate(prompt, generator=...)` | 从图像或文本生成带部件、关节和语义标注的铰接资产草案。 | Articulate Anything、Arti-PG、EmbodiedGen |
| `assets.calibrate(asset, observations=...)` | 用真实观测反演质量、摩擦、刚度、阻尼和接触参数并发布新版本。 | MuJoCo、Genesis、SciPy、Optuna |
| `assets.validate(asset, profile="digital_twin")` | 执行图结构、制品、碰撞、惯量、关节限制和物性来源检查。 | trimesh、CoACD/V-HACD、MuJoCo、Genesis |
| `assets.register(spec)` | 将几何、纹理、语义、关节、碰撞和物性清单发布为不可变资产版本。 | OpenUSD、Pydantic、内容寻址存储 |

实现约定：铰接重建优先选用已发布代码的 ArtGS 系方法；ArtGen 保留为生成类插件，无可复现实现时使用 Articulate Anything/Arti-PG。

## 3. 场景与任务生成

| API | 用途 | 推荐开源实现 |
| --- | --- | --- |
| `scenes.compose(template=..., assets=...)` | 按确定模板组合机器人、物体、传感器、环境和随机化约束。 | OpenUSD、BlenderProc、RoboCasa |
| `scenes.generate(prompt, generator=...)` | 根据场景描述批量生成布局、初始状态、扰动条件和长尾变体。 | ProcTHOR、BlenderProc、EmbodiedGen |
| `tasks.compile(draft, scene=...)` | 把自然语言或结构化草案编译为实体绑定、动作空间和可判定成功条件。 | PDDL、Fast Downward、BehaviorTree.CPP |
| `tasks.generate_demonstrations(task, planner=...)` | 为任务生成运动规划、遥操作或策略回放得到的成功与失败轨迹。 | MoveIt 2/OMPL、MimicGen、ManiSkill |
| `tasks.validate(task, scene=..., backend=...)` | 检查实体引用、信号模式、可达性、碰撞和后端能力是否一致。 | MoveIt 2、MuJoCo、Genesis、Isaac Lab |

## 4. 物理仿真与多模态渲染

| API | 用途 | 推荐开源实现 |
| --- | --- | --- |
| `runs.submit(spec)` | 固定场景、任务、后端、策略、种子和记录信号后提交仿真运行。 | Isaac Lab、Genesis、MuJoCo、ManiSkill |
| `runs.watch(run)` | 连续读取运行状态、进度、质量事件和资源指标。 | OpenTelemetry、Temporal/Ray event stream |
| `runs.wait(run)` | 等待运行完成并返回原子发布的数据集精确引用。 | Temporal、PostgreSQL、MinIO |
| `runs.episodes(run, filters=...)` | 按成功状态、失败类别和随机化实值检索 Episode。 | Apache Arrow、Parquet、DuckDB |
| `datasets.get(ref)` | 读取包含 RGB、Depth、Mask、Event、Pose、点云、关节、接触和力觉模式的数据集清单。 | LeRobot dataset、RLDS、Zarr、MCAP |

渲染适配器使用 BlenderProc 或 Omniverse Replicator，事件相机使用 ESIM，域随机化使用 Isaac Lab、ManiSkill 或 robosuite；Isaac Sim、Genesis 和 MuJoCo 只作为可替换执行后端。

## 5. 数据管理、世界模型与策略训练

| API | 用途 | 推荐开源实现 |
| --- | --- | --- |
| `datasets.validate(dataset, level="deep")` | 校验 Episode 模式、时间轴、来源摘要、质量门和制品完整性。 | Pandera、PyArrow、DuckDB、Great Expectations |
| `datasets.split(dataset, by=..., ratios=...)` | 按资产、场景或真机批次进行防泄漏切分并发布稳定视图。 | scikit-learn、Polars、DuckDB |
| `datasets.export(dataset, format=...)` | 将同一数据集导出为 LeRobot、RLDS、Parquet、Zarr、HDF5 或 rosbag2。 | LeRobot、TensorFlow Datasets/RLDS、PyArrow、Zarr |
| `training.submit(TrainingSpec(trainer="cape"))` | 训练动作条件并行世界模型并输出可复现模型清单。 | DINOv2、PyTorch、timm、FAISS、OpenDW 基线 |
| `runs.submit(RunSpec(policy="cape_planner"))` | 用候选动作和目标状态执行想象 Rollout、反事实比较与规划。 | CAPE 插件、CEM、FAISS、OpenDW |
| `training.submit(TrainingSpec(trainer="vla"))` | 训练视觉语言动作模型并保持观测、动作和机器人本体模式可追踪。 | LeRobot、OpenVLA、PyTorch |
| `training.submit(TrainingSpec(trainer="imitation"))` | 训练行为克隆、ACT 或 Diffusion Policy 等示范学习策略。 | robomimic、LeRobot、Diffusion Policy |
| `training.submit(TrainingSpec(trainer="reinforcement"))` | 在版本化场景任务上训练强化学习策略并保存环境与评测快照。 | Stable-Baselines3、CleanRL、RSL-RL、skrl |
| `training.submit(TrainingSpec(trainer="vq_action_flow"))` | 先以 VQ-VAE 学习离散动作意图，再用 Diffusion 或 Flow Matching 解码连续控制。 | vector-quantize-pytorch、Diffusion Policy、facebookresearch/flow_matching |

实现约定：CAPE 的 DINO 特征、动作条件并行预测、AdaLN/ViT 解码和对比目标封装为独立训练器；在官方代码可用前，使用 DINOv2、PyTorch 与 OpenDW 组成可复现基线。

## 6. 评测与真机反馈闭环

| API | 用途 | 推荐开源实现 |
| --- | --- | --- |
| `evaluations.submit(model=..., matrix=...)` | 在冻结场景任务矩阵上评测成功率、轨迹误差、接触稳定性、耗时和泛化能力。 | Gymnasium、ManiSkill、RoboCasa、SciPy |
| `evaluations.compare(reports, baseline=...)` | 对齐同一评测矩阵并输出差值、置信区间和失败分布。 | SciPy、statsmodels、Polars |
| `deployments.deploy(model=..., robot=...)` | 通过安全门禁把模型部署到机器人工作节点并记录全部输入输出。 | ROS 2、ros2_control、MoveIt 2 |
| `deployments.episodes(deployment)` | 将真机观测、动作、日志、截图和传感器数据发布为真实来源 Episode。 | rosbag2、MCAP、LeRobot dataset |
| `feedback.ingest(source=...)` | 接收仿真、评测或真机失败证据并绑定原运行来源。 | OpenLineage、MLflow、Apache Arrow |
| `feedback.analyze(feedback, analyzer=...)` | 按感知、语义、规划、控制、动力学和接触类别生成失败归因。 | Polars、DuckDB、scikit-learn、SHAP |
| `feedback.plan_regeneration(analysis, budget=...)` | 根据覆盖缺口和失败分布生成数据补采、物性校准或仿真再生成计划。 | Optuna、Ray Tune、Hydra |
| `feedback.execute(plan)` | 把再生成计划编译为普通运行并复用监控、取消和数据发布语义。 | Temporal/Ray、现有运行执行器 |

## 平台公共底座

| API | 用途 | 推荐开源实现 |
| --- | --- | --- |
| `workers.register(descriptor)` | 注册仿真、重建、训练或机器人工作节点的版本和能力声明。 | FastAPI、Pydantic、gRPC |
| `workers.list(requirements=...)` | 查询满足 GPU、后端、传感器和机器人能力约束的工作节点。 | Kubernetes、Ray、NVIDIA Device Plugin |
| `jobs.get(job)` | 读取构建、运行、训练、评测或导出任务的稳定状态。 | Temporal、Ray |
| `jobs.watch(job)` | 从可恢复游标连续读取任务状态、进度和诊断事件。 | Temporal、OpenTelemetry |
| `jobs.cancel(job)` | 以显式原因请求取消任务并保留最终审计记录。 | Temporal、Ray |
| `artifacts.put(payload)` | 上传数据、模型、日志或配置并返回内容寻址引用。 | MinIO、S3 API、fsspec |
| `artifacts.get(ref)` | 按精确摘要读取制品并验证响应长度和媒体类型。 | MinIO、S3 API、fsspec |
| `artifacts.verify(ref)` | 重新计算摘要并报告制品是否完整且未被篡改。 | SHA-256、MinIO、fsspec |
| `lineage.trace(resource)` | 查询资产、任务、运行、Episode、数据集和模型之间的全过程来源。 | OpenLineage、MLflow、PostgreSQL |
| `streams.open(topic)` | 实时传输点云、位姿和设备状态并支持录制与回放。 | ROS 2/DDS、Foxglove、MCAP |
| `visualization.open(resource)` | 在浏览器加载 glTF/USD 派生资产、轨迹和 WebXR 场景。 | three.js、WebXR、Draco、Foxglove |

## 算法接入规则

1. 每种算法实现 `descriptor + capability + execute` 适配器，不修改资产、场景、任务和 Episode 公共模型。
2. 所有输入在调度前解析为带摘要的精确引用，算法输出通过质量门后才发布新资源版本。
3. 训练权重、许可证、数据许可、运行环境和随机种子进入运行清单，不能只记录算法名称。
4. 未公开官方代码的方案算法不得伪装成已集成功能，应使用同能力开源基线并保持插件名和结果来源明确。

## 关键开源项目

- [OMNI-DC](https://github.com/princeton-vl/OMNI-DC)、[CompletionFormer](https://github.com/youmi-zym/CompletionFormer)、[Open3D](https://github.com/isl-org/Open3D)、[OpenUSD](https://github.com/PixarAnimationStudios/OpenUSD)
- [gsplat](https://github.com/nerfstudio-project/gsplat)、[VideoArtGS](https://videoartgs.github.io/)、[FreeArtGS](https://github.com/FreeArtGS/FreeArtGS)、[Articulate Anything](https://github.com/vlongle/articulate-anything)
- [BlenderProc](https://github.com/DLR-RM/BlenderProc)、[MimicGen](https://github.com/NVlabs/mimicgen)、[ManiSkill](https://github.com/haosulab/ManiSkill)、[RoboCasa](https://github.com/robocasa/robocasa)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)、[Genesis](https://github.com/Genesis-Embodied-AI/Genesis)、[MuJoCo](https://github.com/google-deepmind/mujoco)、[ESIM](https://github.com/uzh-rpg/rpg_esim)
- [LeRobot](https://github.com/huggingface/lerobot)、[robomimic](https://github.com/ARISE-Initiative/robomimic)、[Diffusion Policy](https://github.com/real-stanford/diffusion_policy)、[OpenDW](https://github.com/dexmal/opendw)
- [ROS 2](https://github.com/ros2/ros2)、[rosbag2](https://github.com/ros2/rosbag2)、[MCAP](https://github.com/foxglove/mcap)、[OpenLineage](https://github.com/OpenLineage/OpenLineage)
