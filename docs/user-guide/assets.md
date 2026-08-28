# 资产与数字孪生

本模块从**已采集数据包**开始，将 RGB-D、多状态观测和候选重建结果转换为可仿真、可训练、可追溯的铰接资产。数据采集设备和驱动不属于该 API 边界。

## 标准流程

```text
源包预检 → 不可变观测集 → 标准化/相机 → 可选深度补全
          → 铰接重建 → 碰撞与物性 → USD 封装 → 质量验收
```

不要用一个 `method` 同时表示整条流水线。整体 pipeline 负责调度和隔离运行环境，每个阶段的 `AlgorithmSpec` 单独固定方法、版本、权重和配置。

## 1. 预检 ArtModeling 数据包

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

预检要求：

- `real/can/start` 和 `real/can/end` 至少各有一对 `img/<take>/` 与 `depth/<take>_depth.npy`；
- 每个 take 的 RGB 帧从 `0.png` 开始连续编号，数量与对应 NPY 首维一致；
- 可选 `real/cmp/start`、`real/video`、`real/recon`和 `sim`；
- 整个源包不能含符号链接，避免越界导入；
- 缺少标定或深度单位时必须给出明确诊断。

## 2. 发布观测集

```python
source = px.assets.ingest(
    "/data/ArtModeling/cabinet2",
    resource_id="cabinet2.observations",
    version="1.0.0",
    display_name="Cabinet 2 observations",
    importer="artmodeling",
    config={
        "depth_unit": "mm",
        "coordinate_frame": "camera",
        "include_candidates": True,
    },
)
```

导入不修改源目录。每个 RGB 帧、打包深度、PLY、USD 和元数据文件都在流式计算 SHA-256 后写入对象存储；文件在复制期间变化则整次导入失败。

`real/recon` 和 `sim` 中的内容只是**候选制品**，不会因为存在就被视为正确资产。

## 3. 选择阶段方法

| 输入条件 | 建议方法 | 输出 |
| --- | --- | --- |
| 已有可信内外参 | `provided_calibration` | 标准相机模型 |
| 多视角 RGB 但缺相机位姿 | COLMAP | 相机位姿和稀疏点云 |
| RGB + 稀疏/缺损深度 | OMNI-DC | 鲁棒稠密深度 |
| NYUv2 类室内 RGB-D | CompletionFormer | 已有权重的基线深度 |
| 多视角、多铰接状态 | ArtGS 方案 / FreeArtGS 开源基线 | 部件、3D Gaussian 和关节参数 |
| 单目连续视频 | VideoArtGS | 可交互铰接数字孪生 |
| 已有 Mesh 部件 | CoACD + trimesh | 碰撞代理、质量属性 |

RDFC-GAN 保留为论文方法选项，但其官方仓库未提供可直接下载的预训练权重，因此不设为默认深度插件。

## 4. 声明可复现配方

```python
from physrealx import (
    AlgorithmSpec,
    DigitalTwinBuildSpec,
    DigitalTwinRecipe,
    DigitalTwinStage,
    DigitalTwinStep,
)

def step(stage, method, *, checkpoint=None, config=None):
    return DigitalTwinStep(
        stage=stage,
        algorithm=AlgorithmSpec(
            name=method,
            version="1.0.0",
            checkpoint=checkpoint,
            config=config or {},
        ),
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
```

权重应先写入制品存储，然后以 `ArtifactRef` 放入 `AlgorithmSpec.checkpoint`；不要在配方中写可变的文件路径或 `latest`。

## 5. 构建与验收

```python
result = px.assets.build(
    DigitalTwinBuildSpec(
        resource_id="cabinet2",
        version="1.0.0",
        source=source,
        recipe=recipe,
    )
)

asset = px.assets.get(result.asset)
manifest = px.assets.get_build(result.build)
```

构建 pipeline 必须返回与请求一致的资产身份、与配方完全对齐的阶段记录和 `digital_twin` 质量报告。未发布制品、缺失视觉/碰撞表示、无效摩擦关系或错误级质量问题都会阻止发布。

## 资产语义

最终 `AssetSpec` 至少包含：

- 稳定的 link/joint 树，每个非根 link 只有一个父关节；
- 分离的视觉和碰撞表示，且全部几何有精确内容摘要；
- 单位化关节轴、合法限位、阻尼、摩擦及可选力/速度限制；
- 每个 link 的质量、质心、正定惯量、摩擦和恢复系数；
- 每个物性值的 `measured` / `inferred` / `default` 来源、不确定度和置信度。

修改几何、关节、碰撞或物性时必须发布新版本；历史场景和运行通过摘要继续引用原资产。
