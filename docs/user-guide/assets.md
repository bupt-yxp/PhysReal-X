# 资产与数字孪生

本模块从**已采集数据包**开始，将多状态 RGB-D 和已有候选制品转换为可仿真、可训练、可追溯的
铰接资产；相机驱动和采集设备不在平台 API 边界内。

## 标准流程

```text
源包预检 → 不可变观测集 → 坐标/相机 → 可选深度增强 → 几何重建
→ 部件与关节 → 碰撞代理 → 物性标定 → OpenUSD → 质量验收
```

整体 pipeline 只负责编排、隔离和汇总证据。每个阶段绑定一个匹配能力的 `MethodSpec`，因此
COLMAP、OMNI-DC、Ditto、CoACD 和 MuJoCo 可以独立替换，不会彼此冒充。

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

预检确认必需状态、RGB/Depth 帧数、连续编号、NPY 头、符号链接、标定文件和单位声明。
`real/recon` 与 `sim` 中的文件只是候选证据，不会自动变成可信资产。

## 2. 发布观测集

```python
source = px.assets.ingest(
    "/data/ArtModeling/cabinet2",
    resource_id="cabinet2.observations",
    version="1.0.0",
    display_name="Cabinet 2 observations",
    config={
        "depth_unit": "mm",
        "coordinate_frame": "camera",
        "include_candidates": True,
    },
)
```

导入不修改源目录。文件在复制期间发生变化时整次导入失败；成功后每个流和附件都具有精确摘要。

## 3. 安装并选择方法

外部算法包通过 `physrealx.workers` 入口注册独立进程或 OCI worker。先确认可执行状态，再绑定版本与配置：

```python
from physrealx import MethodCapability

print(px.methods.installations(MethodCapability.ARTICULATION_RECOVER))

joint_method = px.methods.bind(
    MethodCapability.ARTICULATION_RECOVER,
    "ditto",
)

if not px.methods.inspect(MethodCapability.ARTICULATION_RECOVER, "ditto").ready:
    raise RuntimeError("Ditto worker 未正确安装")
```

推荐起点：

| 条件 | 方法 | 主要产物 |
| --- | --- | --- |
| 已有可信相机标定 | `provided_calibration` | 规范相机模型 |
| 多视角 RGB 缺位姿 | `colmap` | 位姿与稀疏结构 |
| 缺损 RGB-D | `omni_dc` | 稠密深度及置信度 |
| 已标定 RGB-D | `open3d_fusion` | Mesh/点云 |
| 交互前后点云 | `ditto` | 部件、关节轴、类型与范围 |
| 已分部件 Mesh | `coacd` | 碰撞代理 |
| 多状态动力学观测 | `mujoco_sysid` | 带来源和不确定度的物性 |

RDFC-GAN 无可直接复用的官方权重，不设为默认；ArtGen、PD²GS 等无法核验完整官方实现与许可证
的工作不注册为可执行方法。

## 4. 声明配方

```python
from physrealx import (
    DigitalTwinBuildSpec,
    DigitalTwinRecipe,
    DigitalTwinStage,
    DigitalTwinStep,
    MethodCapability,
)


def bind(stage, capability, name, **config):
    return DigitalTwinStep(
        stage=stage,
        method=px.methods.bind(capability, name, config=config),
    )


recipe = DigitalTwinRecipe(
    name="cabinet_twin",
    version="1.0.0",
    pipeline="digital_twin_worker",
    pipeline_version="1.0.0",
    steps=(
        bind(DigitalTwinStage.NORMALIZATION, MethodCapability.ASSET_NORMALIZE, "normalize"),
        bind(DigitalTwinStage.CAMERA_RECOVERY, MethodCapability.CAMERA_RECOVER, "colmap"),
        bind(
            DigitalTwinStage.GEOMETRY_RECONSTRUCTION,
            MethodCapability.GEOMETRY_RECONSTRUCT,
            "open3d_fusion",
        ),
        DigitalTwinStep(
            stage=DigitalTwinStage.ARTICULATION_RECOVERY,
            method=joint_method,
        ),
        bind(DigitalTwinStage.COLLISION_BUILD, MethodCapability.COLLISION_BUILD, "coacd"),
        bind(
            DigitalTwinStage.PHYSICS_CALIBRATION,
            MethodCapability.PHYSICS_CALIBRATE,
            "mujoco_sysid",
        ),
        bind(DigitalTwinStage.ASSET_PACKAGING, MethodCapability.ASSET_PACKAGE, "openusd"),
        bind(
            DigitalTwinStage.QUALITY_VALIDATION,
            MethodCapability.ASSET_VALIDATE,
            "physics_probe",
        ),
    ),
)
```

深度增强是唯一可省略阶段；其余阶段必须唯一且按固定顺序出现。更换源码、权重、配置或运行镜像
都会产生不同配方证据，不能覆盖历史结果。

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

至少检查：相机重投影误差、深度有效率与边界误差、几何 Chamfer/F-score、关节轴/原点/范围误差、
碰撞近似误差、质量与摩擦拟合残差、穿模率、接触稳定性及多状态回放误差。没有真实参考值的指标
必须标记不可用，不能用渲染图代替定量验收。

最终 `AssetSpec` 保存稳定 link/joint 树、分离的视觉/碰撞几何、单位化关节轴、合法限位以及带
`measured` / `inferred` / `default` 来源、不确定度和置信度的物理参数。任何修改都发布新版本。
