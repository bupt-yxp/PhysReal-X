---
title: 能力与方法
description: 已实现的方法、冻结契约和执行边界
---

# 能力与方法

平台按 `capability + method` 选择实现：`MethodDescriptor` 固定能力、版本、代码来源、许可证和输入输出契约；`MethodSpec` 冻结配置与可选权重。跨阶段流程由现有 `MethodRecipe` 组合，输入只能来自实际发布的资源或制品。

## 安装与运行边界

```bash
python -m pip install -e '.[physical,train,colmap,visual-learning]'
# ArtGS extra 提供已知依赖组合；已有环境由实际 CUDA/扩展探针判断。
python -m pip install --no-build-isolation -e '.[artgs]'
```

`physical` 固定 NumPy、SciPy、CoACD、Trimesh、MuJoCo 和 OpenUSD 的版本，支持 Python 3.10–3.12；所有依赖装入当前项目环境。原生库按需加载，导入 SDK 不初始化引擎、GPU 或工作区。修改后的 ArtGS 源码留在 `algorithms/`，未修改的官方二进制绑定通过 extra 安装。

## 已实现的方法

| 能力 | 方法 | 实现与验收边界 |
| --- | --- | --- |
| `observation.import` | `artmodeling` | 已采集数据预检与不可变导入 |
| `camera.recover` | `colmap` | 自然特征与可选 Marker 辅助相机恢复，保留未解析尺度 |
| `camera.align_metric` | `ransac_umeyama` | RANSAC + 正旋转 Sim(3) 拟合；拒绝退化、错绑证据和留出误差超限 |
| `camera.align_metric` | `usd_depth_scale` | USD 高度辅助深度/稀疏尺度估计，整视角留出检查；明确标记模型条件尺度 |
| `camera.align_metric` | `articulated_surface_icp` | 两状态 RGB-D/运动组到给定 USD 的完整位姿与滑动关节拟合，留出视角检查；当前参考轴为 +X |
| `depth.enhance` | `rgb_guided_fill` | RGB 引导局部小孔补全；显式区分独立深度参考和连续表面遮挡像素自监督，后者保留无效区域 |
| `geometry.reconstruct` | `rgbd_fusion` | CPU 实拍可见表面重建，逐视角绑定与留出距离分布；不补造不可见几何或部件标签 |
| `geometry.reconstruct` | `artgs` | 多状态视觉重建；需要米制相机、通过验证的深度、实际可用的 CUDA/扩展及独立视图 |
| `articulation.recover` | `rigid_pair_fit` | 已对应、已分部件的米制点对拟合单自由度移动/转动关节；独立点集验证，不估计未观察到的限位 |
| `collision.build` | `coacd` | 米制凸分解与 GLB 重读；拒绝非封闭输入、非凸结果和体积误差超限 |
| `physics.calibrate` | `bounded_inverse_dynamics` | 重力补偿的单关节惯量、阻尼和库仑摩擦辨识；检查激励秩、边界、置信区间及独立轨迹改进 |
| `asset.assemble` | `measured_asset_assembly` | 逐 link 几何和实测刚体物性、逐 joint 辨识结果闭包；不从等效关节惯量猜测 link 质量 |
| `asset.assemble` | `reference_asset_assembly` | USD 结构与逐 link 碰撞闭包；全部物性初值和相邻接触排除显式记录为 prior |
| `asset.package` | `openusd` | 自包含 USDZ（含纹理与已绑定高斯模型）；数值复核几何、质量、惯量、关节和单位 |
| `asset.validate` | `mujoco_holdout` | 第三份独立轨迹的真实动力学回放；报告具体位置/速度误差 |
| `simulation.run` | `mujoco` | 通过 `runs.submit` 执行场景、关节控制和同步多相机 RGB/Depth/Mask/Event/点云/位姿及刚体外力/力矩；当前固定各资产根节点 |
| `simulation.run` | `reference_drawer` | 平台契约回归 |
| `render.generate` | `registered_artgs` | CUDA 原生高斯渲染，由同一物理资产的实际关节状态驱动；单目标物体 |
| `policy.inference` | `joint_target` | 与引擎无关的确定性位置/增量目标控制 |
| `policy.inference` | `scripted_drawer` | Reference 后端契约回归 |
| `policy.train` / `policy.inference` | `torch_bc` | 数值行为克隆基线，支持从 MuJoCo Episode 训练并重新评测 |
| `scene.generate` / `scene.compose` | `bounded_layout` / `collision_constraints` | 有界采样、静态包围盒约束与不可行失败 |
| `task.compile` / `task.validate` | `articulated_templates` / `joint_constraints` | 绑定真实关节的开关语言模板与控制边界检查 |
| `trajectory.generate` | `mujoco_joint_trajectory` | 实际执行检查初态、跟踪、接触和终态 |
| `policy.inference` | `joint_sequence` | 显式位置轨迹执行 |
| `policy.train` / `policy.inference` | `act` | 官方 LeRobot CVAE/Transformer，完整动作块执行，重载权重 |
| `dataset.ingest` / `dataset.snapshot` | `episode_dataset` / `episode_split` | 真实数组校验、核心快照、按来源分组 |
| `world_model.train` / `world_model.rollout` | `dino_wm` | 官方 DINOv2/因果 ViT，记录残差/解码器适配，未来状态/图像与反事实 |
| `telemetry.record` / `failure.analyze` | `episode_telemetry` / `trajectory_diagnostics` | 不可变轨迹指标与症状诊断 |
| `feedback.build_dataset` | `failure_episodes` | 保留失败标签、输出同初态再生成计划 |

```python
from physrealx import MethodCapability

installation = px.methods.inspect(MethodCapability.COLLISION_BUILD, "coacd")
assert installation.ready, installation.detail
```

安装目录中的方法有实现并不意味着本机依赖齐全。`px.methods.installations()` 返回当前真实探针结果；仿真后端也遵守同一规则。`READY` 仅表示可执行条件成立，不代表已有合格实测结果。

## 输入与发布

`enhance_depth`、`articulate`、`calibrate_physics`、`assemble`、`package` 和 `validate_physics` 的阶段证据统一使用 `ArtifactRef`；只有观测集、资产、场景、任务、数据集、模型等已登记领域资源使用 `VersionedRef`。相机统一采用 `streams → components → images`，不保留旧的顶层 `components` 形式。

服务固定输出名、Schema、媒体类型和大小上限。发布 `OperationManifest` 前检查实际 JSON 的有限值、重复键与 Schema，以及 ZIP 的路径、解压大小和 CRC。任何失败均不发布成功操作；已完成的独立上游操作可以复用。

来源 revision、依赖版本、资源版本和平台 `artifact_id` 各自承担不同职责；业务接口不以文件哈希替代资源身份。每个项目实例拥有自己的注册表和存储。

## 尚未接入的扩展

Isaac Sim、Genesis、学习式场景生成、全场景高斯合成、VLA 训练及 ROS 2 硬件适配器仍需独立实现和验收。现有协议允许接入这些能力，候选名称不会被注册为成功方法。真实部署还需要节点 Agent、硬件驱动和现场证据。

`assets.import_reference` 直接从已导入 USD 提取参考网格与关节；`assets.prepare_rgbd` 将源序列解码为绑定视角的阶段制品，复用现有制品库和方法服务。

米制、深度、物性、USD 序列化与物理轨迹验收是不同门禁。`cabinet2` 缺少的实测证据不能由历史训练日志或合成测试夹具代替。

`camera.align_metric:articulated_surface_icp` 在米制 RGB-D 与 ArtGS 运动组基础上，将两状态滑动物体配准到给定 USD，使用独立留出视角检查；当前明确支持参考 +X 滑动轴。`render.generate:registered_artgs` 读取同一物理资产的实际关节轨迹与配准信息，生成该物体的高斯 RGB、透明度和米制期望深度。它们不增加独立计量或未观测关节运动证据。
