# 真实资产接入与数字孪生

本模块从**已经完成采集的数据包**开始，完成不可变导入、相机与尺度恢复、深度增强、视觉几何重建以及部件和关节恢复。相机驱动、硬件触发和在线采集不属于平台 API。

## 完成目标

```text
采集目录 → 结构预检 → 观测集 → 相机恢复 → 米制对齐
→ 深度增强 → 几何重建 → 部件与关节模型
```

输出是带米制坐标、视觉几何和可动结构的数字孪生候选。碰撞体、质量、摩擦和阻尼由[物理可信资产构建](physical-assets.md)处理。

## 1. 检查采集数据

输入可以包含 RGB、Depth、Event、点云、位姿、力觉和机器人状态。导入前至少确认：

- 物体、状态与采集批次具有稳定标识；
- RGB/Depth 帧可明确配对，帧 ID 没有重复；
- 时间戳、单位、坐标系和深度无效值规则有记录；
- 标定、marker、控制量和力觉证据作为独立附件保留；
- 历史 Mesh、USD 和人工参数标记为候选结果，而非实测真值；
- 目录没有越界链接，也没有仍在写入的文件。

```python
from physrealx import PhysRealX

px = PhysRealX.open(".physrealx")
inspection = px.assets.inspect_source(
    "/data/ArtModeling/cabinet2",
    importer="artmodeling",
)

for issue in inspection.issues:
    print(issue.severity, issue.code, issue.path)

if not inspection.ingestible:
    raise RuntimeError("数据包未通过预检")
```

ArtModeling 导入器识别 `real/can/start`、`real/can/end`、`real/cmp`、`real/video`、`real/recon` 与 `sim`，但不会猜测深度尺度或自动采信历史重建。

## 2. 发布不可变观测集

```python
observations = px.assets.ingest(
    "/data/ArtModeling/cabinet2",
    resource_id="cabinet2.observations",
    version="1.0.0",
    display_name="Cabinet 2 observations",
    importer="artmodeling",
    config={
        "depth_unit": "native_u16_unscaled",
        "coordinate_frame": "camera_unregistered",
        "include_candidates": False,
    },
)
```

这里如实声明深度尚未定尺、相机尚未注册。导入过程复制文件并检查复制期间的大小与源文件
状态；任一文件发生变化，整次发布失败。`ObservationSetManifest` 保存帧制品、媒体类型、大小、
导入器版本和配置。

## 3. 检查重建方法

```python
from physrealx import MethodCapability

required = (
    MethodCapability.CAMERA_RECOVER,
    MethodCapability.CAMERA_ALIGN_METRIC,
    MethodCapability.DEPTH_ENHANCE,
    MethodCapability.GEOMETRY_RECONSTRUCT,
    MethodCapability.ARTICULATION_RECOVER,
)

for capability in required:
    print(capability.value, px.methods.installations(capability))
```

`ready=True` 同时要求方法身份、源码 revision 或权重制品、项目环境和真实探针一致。当前发行版
在这条链中只提供 `artgs`；相机、米制对齐、深度和关节方法必须完成项目内集成后才会出现在结果中。

## 4. 恢复相机与米制尺度

```python
camera_operation = px.assets.recover_cameras(
    observations,
    method=camera_method,
    config=camera_config,
)

camera_rig = next(
    output.artifact
    for output in camera_operation.outputs
    if output.name == "camera_rig_calibration"
)

metric_operation = px.assets.align_cameras_metric(
    camera_rig,
    metric_alignment_evidence,
    method=metric_alignment_method,
)
```

纯 RGB COLMAP 重建只确定到 Sim(3)，必须保持 `scale_status="unresolved"`。`MetricAlignmentEvidence/1` 应提供 marker 家族、实际尺寸和布局，或等价的独立米制 3D 对应；拟合点和留出点必须分离。只有内点率、尺度稳定性和留出误差全部通过，才能发布米制相机。

## 5. 增强深度与重建几何

```python
depth_operation = px.assets.enhance_depth(
    calibrated_rgbd_ref,
    method=depth_method,
)
```

深度增强必须保留原始有效像素，并使用独立留出像素检查 RMSE、MAE、REL、覆盖率和边界误差。结果没有改善时，应保留原深度并使增强门禁失败。

内嵌 ArtGS 接收两状态 RGBA、刚体相机位姿和已通过门禁的米制增强深度。平台不信任目录中
预先存在的深度 PNG，而是按显式 frame 绑定读取 `EnhancedDepthBundle/1` 的 float32 米制 NPY，
确定性转换为 ArtGS 所需的 uint16 毫米 PNG，并把每帧制品引用写入输入清单：

```python
from physrealx import package_artgs_input

artgs_source = package_artgs_input(
    "/data/prepared/cabinet2",
    scene_id="cabinet2",
    camera_rig=metric_camera_rig,
    enhanced_depth=enhanced_depth_bundle,
    # 由本次预处理清单读取，必须精确覆盖全部训练与留出视图。
    depth_frame_ids=frame_bindings,
    num_slots=3,
    artifacts=px.artifacts,
)

geometry_operation = px.assets.reconstruct_geometry(
    artgs_source,
    method="artgs",
    config={"seed": 0},
)
```

输入必须同时包含 `transforms_train_start/end.json`、两状态 RGBA 和独立
`transforms_test_start/end.json`；训练视图不能兼作发布验收。只有显式设置
`allow_training_view_evaluation=True` 才允许实验运行，此时报告固定为
`TRAINING_VIEWS_ONLY`，不能进入可训练资产门禁。输出为每 link 视觉 Mesh、Gaussian、形变权重和
`GeometryReconstructionReport/1`；它不替代关节、碰撞或物性验证。

## 6. 恢复部件与关节

```python
articulation_operation = px.assets.articulate(
    paired_interaction_pointcloud_ref,
    method=articulation_method,
)
```

Ditto 输入必须是同一米制对象坐标系中的交互前后点云，并带相机、掩码、运动量和留出点。输出关节轴需要单位化，预测运动必须在留出点上通过误差门禁。单次前后状态通常不足以确定完整限位，应由多状态轨迹或独立测量补充。

## 方法与产物

| 阶段 | 候选实现 | 主要产物 |
| --- | --- | --- |
| 数据导入 | `artmodeling` | `ObservationSetManifest` |
| 相机恢复 | `colmap` | 相机、稀疏重建和恢复报告 |
| 米制对齐 | `metric_alignment` | 米制相机与留出报告 |
| 深度增强 | `omni_dc` | 增强深度和质量报告 |
| 几何重建 | `artgs` | 每 link 视觉 Mesh、Gaussian、形变权重与留出报告 |
| 关节恢复 | `ditto` | `ArticulationModel/1` 与恢复报告 |

除 `artmodeling` 与 `artgs` 外，这些名称用于说明推荐算法映射，不表示当前发行版已经注册对应适配器。

## cabinet2 数据边界

原始包可以完成结构预检、RGB-D 配对和 ArtModeling 导入，但缺少可核验的 Marker 尺寸与布局、
物性辨识轨迹和真机 Episode。平台必须在对应阶段返回 `BLOCKED`，不能用候选重建或工程先验
补齐这些科学证据。

历史实验实际执行 ArtGS 35,000 次优化并生成 Gaussian 与部件 Mesh，但当时没有米制深度绑定、
独立留出视图和关节真值，因此仍只是训练视图算法证据，不能冒充当前 `artgs` 方法的通过结果。

![ArtGS 训练视图对比](../assets/images/experiments/cabinet2-artgs-comparison.png)

## 输出与验收

- 源文件清单、模态配对、单位与坐标声明完整；
- 相机注册率、重投影误差、轨迹长度和稀疏点数达标；
- 米制尺度置信区间、RANSAC 内点率及留出 RMSE 达标；
- 深度覆盖率、RMSE/REL、边界误差和无效值比例达标；
- 几何 Chamfer、F-score、完整率和尺度误差达标；
- 部件非退化，关节轴、原点及留出运动重放误差达标。

没有参考真值的指标必须标为不可用。下一步进入[物理可信资产构建](physical-assets.md)。
