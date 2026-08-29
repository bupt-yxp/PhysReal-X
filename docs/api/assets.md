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
    config={"depth_unit": "mm", "coordinate_frame": "camera"},
)
```

### `assets.get_observations` / `assets.list_observations`

```python
px.assets.get_observations(ref: VersionedRef) -> ObservationSetManifest
px.assets.list_observations() -> list[VersionedRef]
```

## 方法目录

### `methods.list` / `methods.get` / `methods.bind`

```python
px.methods.list(capability=None) -> tuple[MethodDescriptor, ...]
px.methods.get(capability, name) -> MethodDescriptor
px.methods.bind(capability, name, *, checkpoint=None, config=None) -> MethodSpec
```

扩展包安装并注册方法后，用户按能力选择实现。`bind` 会冻结完整方法描述符、配置和可选权重
摘要；未写入本工作区制品存储的权重会被拒绝。

## 数字孪生构建

### `assets.build`

```python
px.assets.build(spec: DigitalTwinBuildSpec) -> DigitalTwinBuildResult
```

`DigitalTwinBuildSpec` 分离整体 worker 与逐阶段方法。每个阶段只接受匹配的
`MethodCapability`，不能用一个 `method` 指代整条链路。

| 阶段 | 所需能力 | 典型方法 |
| --- | --- | --- |
| `normalization` | `asset.normalize` | 单位、坐标、命名和状态对齐 |
| `camera_recovery` | `camera.recover` | COLMAP 或可信现成标定 |
| `depth_enhancement`（可选） | `depth.enhance` | OMNI-DC、CompletionFormer |
| `geometry_reconstruction` | `geometry.reconstruct` | Open3D TSDF/点云融合 |
| `articulation_recovery` | `articulation.recover` | Ditto、ScrewSplat |
| `collision_build` | `collision.build` | CoACD |
| `physics_calibration` | `physics.calibrate` | MuJoCo system identification |
| `asset_packaging` | `asset.package` | OpenUSD |
| `quality_validation` | `asset.validate` | 结构、碰撞、动力学与回放门禁 |

```python
from physrealx import (
    DigitalTwinBuildSpec,
    DigitalTwinRecipe,
    DigitalTwinStage,
    DigitalTwinStep,
    MethodCapability,
)


def step(stage, capability, name, *, checkpoint=None, config=None):
    return DigitalTwinStep(
        stage=stage,
        method=px.methods.bind(
            capability,
            name,
            checkpoint=checkpoint,
            config=config,
        ),
    )


recipe = DigitalTwinRecipe(
    name="cabinet_twin",
    version="1.0.0",
    pipeline="digital_twin_worker",
    pipeline_version="1.0.0",
    steps=(
        step(DigitalTwinStage.NORMALIZATION, MethodCapability.ASSET_NORMALIZE, "normalize"),
        step(DigitalTwinStage.CAMERA_RECOVERY, MethodCapability.CAMERA_RECOVER, "colmap"),
        step(
            DigitalTwinStage.DEPTH_ENHANCEMENT,
            MethodCapability.DEPTH_ENHANCE,
            "omni_dc",
        ),
        step(
            DigitalTwinStage.GEOMETRY_RECONSTRUCTION,
            MethodCapability.GEOMETRY_RECONSTRUCT,
            "open3d_fusion",
        ),
        step(
            DigitalTwinStage.ARTICULATION_RECOVERY,
            MethodCapability.ARTICULATION_RECOVER,
            "ditto",
        ),
        step(DigitalTwinStage.COLLISION_BUILD, MethodCapability.COLLISION_BUILD, "coacd"),
        step(
            DigitalTwinStage.PHYSICS_CALIBRATION,
            MethodCapability.PHYSICS_CALIBRATE,
            "mujoco_sysid",
        ),
        step(DigitalTwinStage.ASSET_PACKAGING, MethodCapability.ASSET_PACKAGE, "openusd"),
        step(
            DigitalTwinStage.QUALITY_VALIDATION,
            MethodCapability.ASSET_VALIDATE,
            "physics_probe",
        ),
    ),
)

result = px.assets.build(
    DigitalTwinBuildSpec(
        resource_id="cabinet2",
        version="1.0.0",
        source=source,
        recipe=recipe,
    )
)
```

运行前平台会核对 pipeline 版本、每个已安装方法的完整描述符和权重摘要。运行结果必须逐阶段
返回相同 `MethodSpec`、时间、指标和已发布制品；阶段缺失、换方法或返回可变文件均会失败。

相同资源、版本、源摘要和配方重复提交时返回原结果；相同身份对应不同输入时拒绝覆盖。

### `assets.get_build` / `assets.list_builds`

```python
px.assets.get_build(ref: VersionedRef) -> DigitalTwinBuildManifest
px.assets.list_builds() -> list[VersionedRef]
```

## 资产发布与校验

```python
px.assets.register(spec: AssetSpec) -> VersionedRef
px.assets.get(ref: VersionedRef) -> AssetSpec
px.assets.list() -> list[VersionedRef]
px.assets.validate(
    asset: AssetSpec | VersionedRef,
    *,
    profile: AssetValidationProfile = AssetValidationProfile.DIGITAL_TWIN,
) -> AssetValidationReport
```

| Profile | 检查 |
| --- | --- |
| `schema` | 标识、坐标、link/joint 图和参数范围 |
| `simulation` | 增加制品完整性、视觉/碰撞表示和摩擦约束 |
| `digital_twin` | 增加物性来源、置信度和关节执行限制 |

只有无 `error` 的候选资产才能发布；`warning` 保留在构建清单中。

## 主要错误

- `SourcePackageError`：数据包缺项、帧数冲突、单位不明或包含符号链接；
- `AssetBuildError`：pipeline 身份、阶段证据、制品或质量门不符合契约；
- `ArtifactIntegrityError`：导入期间文件变化或存储制品摘要不符；
- `ResourceConflictError`：资源身份或已注册方法对应了不同内容。
