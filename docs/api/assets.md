# 资产 API

资产服务从已采集数据包开始，发布观测集、构建数字孪生并校验可仿真资产；不控制相机或采集设备。

## 源数据预检

### `assets.inspect_source`

```python
px.assets.inspect_source(
    source: str | Path,
    *,
    importer: str = "artmodeling",
) -> SourceInspection
```

只读检查目录结构、模态、状态帧数、RGB-D 对齐、标定缺失和不安全链接。内置
`artmodeling` 方法将 `real/can/start`、`real/can/end` 作为必需重建状态，将
`real/cmp`、`real/video`、`real/recon` 和 `sim` 作为可选证据或候选制品。

## 观测集导入

### `assets.ingest`

```python
px.assets.ingest(
    source: str | Path,
    *,
    resource_id: str,
    version: str,
    display_name: str,
    importer: str = "artmodeling",
    config: Mapping[str, JSONValue] | None = None,
) -> VersionedRef
```

文件经流式 SHA-256 校验后写入内容寻址存储，再原子发布不可变 `ObservationSetManifest`。
ArtModeling 导入必须显式提供深度单位和坐标系，平台不猜测尺度或 frame。

```python
source = px.assets.ingest(
    "/data/ArtModeling/cabinet2",
    resource_id="cabinet2.observations",
    version="1.0.0",
    display_name="Cabinet 2 observations",
    config={
        "depth_unit": "native_u16_unscaled",
        "coordinate_frame": "camera_unregistered",
        "include_candidates": False,
    },
)
```

这组值如实保留原始数据尚未定尺、尚未注册的状态；只有独立米制证据和相机对齐门禁通过后，
派生资源才能声明 `m` 及统一对象坐标系。

### `assets.get_observations` / `assets.list_observations`

```python
px.assets.get_observations(ref: VersionedRef) -> ObservationSetManifest
px.assets.list_observations() -> list[VersionedRef]
```

### `assets.recover_cameras`

```python
px.assets.recover_cameras(
    observations: VersionedRef,
    *,
    method: str | MethodSpec,
    config: Mapping[str, JSONValue] | None = None,
) -> OperationManifest
```

从观测集的 RGB 制品闭包执行 COLMAP 特征提取、时序匹配、稀疏重建和模型转换，返回
`CameraRigCalibration/1`、`SparseReconstructionBundle/1` 与 `CameraRecoveryReport/1`。
`execution_device` 可设为 `cpu`、`cuda` 或 `auto`；只有明确的 CUDA/GPU 故障且
`allow_cpu_fallback=true` 时才重建干净 CPU 工作区重试。纯单目 COLMAP 结果一律标记
`scale_status="unresolved"`，不根据深度单位猜测米制尺度。

### `assets.align_cameras_metric`

```python
px.assets.align_cameras_metric(
    camera_rig: ArtifactRef,
    evidence: ArtifactRef,
    *,
    method: str | MethodSpec,
    config: Mapping[str, JSONValue] | None = None,
) -> OperationManifest
```

使用 `MetricAlignmentEvidence/1` 中的独立米制 3D–3D 对应解析 COLMAP
Sim(3) 模糊性；只有 RANSAC 内点、尺度稳定性和留出误差全部通过才发布米制相机标定。
证据格式与 marker 证据边界见[米制相机与重建对齐](metric-alignment-worker.md)。

## 方法目录

### `methods.list` / `methods.get` / `methods.bind`

```python
px.methods.list(capability=None) -> tuple[MethodDescriptor, ...]
px.methods.get(capability, name) -> MethodDescriptor
px.methods.bind(capability, name, *, checkpoint=None, config=None) -> MethodSpec
```

扩展包安装并注册方法后，用户按能力选择实现。`bind` 会冻结完整方法描述符、配置和可选权重
摘要；未写入本工作区制品存储的权重会被拒绝。

## 数字孪生配方执行

数字孪生是由多个独立 worker 组成的有向配方，不存在可跳过阶段证据的“整体构建
worker”。每个 `RecipeStep` 必须绑定一个可执行的进程或容器 `MethodSpec`，且显式声明输入
来源和输出契约。

| 阶段 | 所需能力 | 典型方法 |
| --- | --- | --- |
| `normalization` | `asset.normalize` | 单位、坐标、命名和状态对齐 |
| `camera_recovery` | `camera.recover` | COLMAP 或可信现成标定 |
| `metric_alignment` | `camera.align_metric` | 独立度量证据与留出验收 |
| `depth_enhancement`（可选） | `depth.enhance` | OMNI-DC、CompletionFormer |
| `geometry_reconstruction` | `geometry.reconstruct` | Open3D TSDF/点云融合 |
| `articulation_recovery` | `articulation.recover` | Ditto、ScrewSplat |
| `collision_build` | `collision.build` | CoACD |
| `physics_calibration` | `physics.calibrate` | 实测证据驱动的单关节有界系统辨识 |
| `asset_packaging` | `asset.package` | OpenUSD |
| `quality_validation` | `asset.validate` | 结构、碰撞、动力学与回放门禁 |

```python
from physrealx import (
    CompletedStepOutput,
    ExternalMethodRecipe,
    InitialRecipeInput,
    MethodCapability,
    OperationInput,
    OperationOutputRequirement,
    RecipeExecutionSpec,
    RecipeFinalOutput,
    RecipeInputBinding,
    RecipeStep,
)

recipe = ExternalMethodRecipe(
    name="cabinet2.metric-camera",
    version="1.0.0",
    steps=(
        RecipeStep(
            name="camera",
            method=px.methods.bind(MethodCapability.CAMERA_RECOVER, "colmap"),
            inputs=(
                RecipeInputBinding(
                    name="observations",
                    contract="ObservationSetManifest/1",
                    source=InitialRecipeInput(input_name="capture"),
                ),
            ),
            outputs=(
                OperationOutputRequirement(
                    name="camera_recovery_report",
                    contract="CameraRecoveryReport/1",
                ),
                OperationOutputRequirement(
                    name="camera_rig_calibration",
                    contract="CameraRigCalibration/1",
                ),
                OperationOutputRequirement(
                    name="sparse_reconstruction_bundle",
                    contract="SparseReconstructionBundle/1",
                ),
            ),
        ),
        RecipeStep(
            name="metric_alignment",
            method=px.methods.bind(
                MethodCapability.CAMERA_ALIGN_METRIC,
                "metric_alignment",
            ),
            inputs=(
                RecipeInputBinding(
                    name="camera_rig",
                    contract="CameraRigCalibration/1",
                    source=CompletedStepOutput(
                        step="camera",
                        output="camera_rig_calibration",
                    ),
                ),
                RecipeInputBinding(
                    name="evidence",
                    contract="MetricAlignmentEvidence/1",
                    source=InitialRecipeInput(input_name="metric_evidence"),
                ),
            ),
            outputs=(
                OperationOutputRequirement(
                    name="camera_rig_calibration",
                    contract="CameraRigCalibration/1",
                ),
                OperationOutputRequirement(
                    name="metric_alignment_report",
                    contract="MetricAlignmentReport/1",
                ),
            ),
        ),
    ),
    outputs=(
        RecipeFinalOutput(
            name="metric_camera_rig",
            source=CompletedStepOutput(
                step="metric_alignment",
                output="camera_rig_calibration",
            ),
        ),
    ),
)

result = px.recipes.execute(
    RecipeExecutionSpec(
        resource_id="cabinet2.metric-camera",
        version="1.0.0",
        recipe=recipe,
        inputs=(
            OperationInput(name="capture", resource=source),
            OperationInput(name="metric_evidence", artifact=metric_alignment_evidence),
        ),
    )
)
```

示例只连接已定义严格契约的“相机恢复 → 米制对齐”；`metric_alignment_evidence`
必须是先行发布的 `MetricAlignmentEvidence/1` 制品。完整资产链继续以相同方式连接
深度增强、几何重建、关节恢复、碰撞代理、物性辨识、
OpenUSD 打包和质量验收。执行前会校验全部 worker 身份、环境、权重和初始证据；任一步失败时
不发布 `RecipeExecutionManifest`。每个 `RecipeInputBinding.contract` 必须与当前方法的输入契约及
上游实际输出同时一致，禁止通过重标签跨过不兼容阶段。成功后每步对应一个可查的
`OperationManifest`，相同规格重试复用同一不可变执行证据。

## 独立资产处理 API

需要单独运行或比较某个阶段时，使用与能力一一对应的服务：

```python
depth = px.assets.enhance_depth(calibrated_rgbd, method="omni_dc")
articulation = px.assets.articulate(interaction_pair, method="ditto")
collision = px.assets.build_collision(
    visual_mesh,
    method="coacd",
    config={"source_unit": "m", "link": "drawer"},
)
identified = px.assets.calibrate_physics(
    system_identification_evidence,
    method="physrealx_mujoco_sysid_adapter",
)
package = px.assets.package(asset_assembly, method="openusd")
```

| API | 不可变输入 | 精确输出契约 |
| --- | --- | --- |
| `assets.enhance_depth` | `rgbd` 资源：`CalibratedRGBDBundle/1` | `EnhancedDepthBundle/1` ZIP；`DepthEnhancementReport/1` JSON |
| `assets.articulate` | `pair` 资源：`PairedInteractionPointCloudBundle/1` | `ArticulationModel/1` JSON；`ArticulationRecoveryReport/1` JSON |
| `assets.build_collision` | `asset` 制品：`VisualMesh/1` | `CollisionGeometry/1` GLB；`CollisionBuildReport/1` JSON |
| `assets.calibrate_physics` | `evidence` 资源：`JointSystemIdentificationEvidence/1` | `IdentifiedJointPhysics/1` JSON；`PhysicsIdentificationReport/1` JSON |
| `assets.package` | `asset` 资源：`AssetAssemblyBundle/1` | `UsdAssetBundle/1` USDC；`AssetPackageManifest/1` JSON；`UsdRoundTripValidationReport/1` JSON |

服务拥有输入名、资源/制品传输形式、Schema、输出名、媒体类型和大小上限；
调用方不能用 `outputs` 重定义公共能力。所选 `MethodDescriptor` 必须与该契约完全
相符，否则在启动 worker 前失败。

`calibrated_rgbd` 必须已经具备米制相机标定和独立留出深度；`interaction_pair` 必须包含
同一米制对象坐标系下的交互前后点云；物性辨识输入必须包含实测时序、执行器作用量和隔离的
训练/留出采集。`IdentifiedJointPhysics/1` 只表示单关节辨识结果；资产组装阶段必须将每个关节
结果与独立实测的 link 质量、质心和惯量合并为完整 `IdentifiedPhysics/1` 制品。OpenUSD 打包输入
同时绑定视觉网格、碰撞网格、已识别关节、完整物性制品和证据闭包。

CoACD 不负责估计尺度。调用方必须用 `source_unit="m"` 明确声明上游视觉 Mesh 已完成米制定尺，
并用 `link` 标识其所属刚体。结果报告绑定输入 Mesh 与实际生成 GLB 的 SHA-256、大小和媒体类型；
缺少单位或 link 时 worker 会拒绝执行。
核心另外定义 `asset.assemble` 就绪能力，但不提供伪默认实现：它必须将米制视觉几何、每 link 碰撞代理、关节模型、
全部 body 物性与每个 joint 辨识结果闭包为 `AssetAssemblyBundle/1`。实现并注册该 worker 之前，就绪检查会如实返回 `BLOCKED`。
任一前提缺失时对应 worker 直接失败，不从已有仿真文件或默认参数回填。

这些方法返回 `OperationManifest`，而不是未校验的文件路径。每份 manifest 包含精确
`MethodSpec`、输入摘要、输出制品、指标与时间证据，可通过 `px.operations.get(...)` 重新读取。

`AssetSpec.evidence_state` 明确区分 `prior → identified → validated`；`identified` 和 `validated`
必须携带内容寻址证据。估计参数不能写成实测，训练变体也不能覆盖特定实物的数字孪生。

## 资产发布与校验

```python
px.assets.register(spec: AssetSpec) -> VersionedRef
px.assets.get(ref: VersionedRef) -> AssetSpec
px.assets.list() -> list[VersionedRef]
px.assets.validate(
    asset: AssetSpec | VersionedRef,
    *,
    profile: AssetValidationProfile | str = AssetValidationProfile.DIGITAL_TWIN,
) -> AssetValidationReport
```

| Profile | 检查 |
| --- | --- |
| `schema` | 标识、坐标、link/joint 图和参数范围 |
| `simulation` | 增加制品完整性、视觉/碰撞表示和摩擦约束 |
| `digital_twin` | 增加物性来源、置信度和关节执行限制 |
| `trainable` | 将默认物性、零置信度、不完整关节限制和缺少留出验证证据升级为错误 |

只有无 `error` 的候选资产才能发布；`warning` 保留在构建清单中。

## 主要错误

- `SourcePackageError`：数据包缺项、帧数冲突、单位不明或包含符号链接；
- `MethodExecutionError`：worker 退出异常、超时或返回了不符合契约的证据；
- `ArtifactIntegrityError`：导入期间文件变化或存储制品摘要不符；
- `ResourceConflictError`：资源身份或已注册方法对应了不同内容。
