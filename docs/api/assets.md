# 资产 API

资产服务处理已采集数据包、数字孪生构建和版本化资产；不提供相机、力传感器或采集设备控制接口。

## 源数据预检

### `assets.inspect_source`

```python
px.assets.inspect_source(
    source: str | Path,
    *,
    importer: str = "artmodeling",
) -> SourceInspection
```

只读检查目录结构、模态、状态帧数、RGB-D 对齐、标定缺失和不安全链接，不写入工作区。

`ArtModelingImporter` 将 `real/can/start` 和 `real/can/end` 视为必需重建状态，将 `real/cmp/start` 和 `real/video` 视为可选验证或时序观测。

## 已采集数据导入

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

将通过预检的文件流式写入内容寻址存储，并发布不可变 `ObservationSetManifest`。ArtModeling 导入必须显式提供 `depth_unit` 和 `coordinate_frame`，不允许平台猜测尺度或坐标语义。

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

读取观测集清单或列出已发布版本，清单中每个原始文件都保留 SHA-256 摘要、字节数和媒体类型。

## 数字孪生构建

### `assets.build`

```python
px.assets.build(spec: DigitalTwinBuildSpec) -> DigitalTwinBuildResult
```

`DigitalTwinBuildSpec` 把整体 pipeline 和每个阶段的方法分开；`ArtGS` 只能出现在铰接重建阶段，不能代替整条流水线。

| 阶段 | 职责 | 可选方法 / 开源基线 |
| --- | --- | --- |
| `normalization` | 解码、单位、坐标、命名和状态对齐 | OpenCV、Open3D |
| `camera_recovery` | 使用已知标定或恢复相机位姿 | COLMAP、VGGT |
| `depth_enhancement` | 可选的 RGB-D 深度补全 | OMNI-DC、CompletionFormer |
| `articulated_reconstruction` | 恢复几何、部件、关节轴与范围 | ArtGS（方案方法）、VideoArtGS、FreeArtGS |
| `physics_enrichment` | 生成碰撞体并添加带来源的物性 | CoACD、trimesh、MuJoCo |
| `simulation_packaging` | 封装视觉、碰撞、关节和物性 | OpenUSD |
| `quality_validation` | 执行结构、制品和仿真质量门 | PhysReal-X validator、MuJoCo |

除 `depth_enhancement` 外，上述阶段必须按固定顺序出现且不得重复。每个 `AlgorithmSpec` 必须记录名称、语义版本、配置与可选权重摘要。

```python
from physrealx import (
    AlgorithmSpec,
    DigitalTwinBuildSpec,
    DigitalTwinRecipe,
    DigitalTwinStage,
    DigitalTwinStep,
)


def step(stage, name):
    return DigitalTwinStep(
        stage=stage,
        algorithm=AlgorithmSpec(name=name, version="1.0.0"),
    )


recipe = DigitalTwinRecipe(
    name="cabinet_twin",
    version="1.0.0",
    pipeline="artgs_openusd",
    pipeline_version="1.0.0",
    steps=(
        step(DigitalTwinStage.NORMALIZATION, "artmodeling_rgbd"),
        step(DigitalTwinStage.CAMERA_RECOVERY, "colmap"),
        step(DigitalTwinStage.DEPTH_ENHANCEMENT, "omni_dc"),
        step(DigitalTwinStage.ARTICULATED_RECONSTRUCTION, "artgs"),
        step(DigitalTwinStage.PHYSICS_ENRICHMENT, "coacd_mujoco"),
        step(DigitalTwinStage.SIMULATION_PACKAGING, "openusd"),
        step(DigitalTwinStage.QUALITY_VALIDATION, "physrealx_validator"),
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

相同资源、版本、源摘要和配方重复提交时直接返回已发布结果；同一身份对应不同输入时拒绝覆盖。

### `assets.get_build` / `assets.list_builds`

```python
px.assets.get_build(ref: VersionedRef) -> DigitalTwinBuildManifest
px.assets.list_builds() -> list[VersionedRef]
```

构建清单保留输入观测摘要、完整配方、逐阶段耗时/指标/制品、最终资产和质量报告。

## 资产发布与校验

### `assets.register` / `assets.get` / `assets.list`

```python
px.assets.register(spec: AssetSpec) -> VersionedRef
px.assets.get(ref: VersionedRef) -> AssetSpec
px.assets.list() -> list[VersionedRef]
```

直接发布、读取或列出结构化资产；已发布的 `resource_id@version` 不可覆盖。

### `assets.validate`

```python
px.assets.validate(
    asset: AssetSpec | VersionedRef,
    *,
    profile: AssetValidationProfile = AssetValidationProfile.DIGITAL_TWIN,
) -> AssetValidationReport
```

| Profile | 检查 |
| --- | --- |
| `schema` | 标识、坐标、图结构与参数范围 |
| `simulation` | 增加制品完整性、每 link 视觉/碰撞表示和摩擦约束 |
| `digital_twin` | 增加物性来源、置信度和关节执行限制检查 |

只有无 `error` 的候选资产才能发布；`warning` 必须保留在构建清单中。

## 主要错误

- `SourcePackageError`：数据包缺项、帧数冲突、单位不明或包含符号链接；
- `AssetBuildError`：pipeline 输出身份、阶段证据、制品或质量门不符合契约；
- `ArtifactIntegrityError`：文件导入期间变化或存储制品与摘要不符；
- `ResourceConflictError`：不可变资源身份已对应其他内容。
