# 资产与数字孪生

本模块从**已采集数据包**开始，将多状态 RGB-D 和已有候选制品转换为可仿真、可训练、可追溯的
铰接资产；相机驱动和采集设备不在平台 API 边界内。

## 标准流程

```text
源包预检 → 不可变观测集 → 未定尺相机恢复 → 米制对齐 → 深度增强 → 几何重建
→ 部件与关节 → 碰撞代理 → 单关节物性辨识 → 完整资产组装 → OpenUSD → 质量验收
```

平台用显式有向配方负责编排、隔离和汇总证据。每个阶段绑定一个匹配能力的 `MethodSpec`，因此
COLMAP、OMNI-DC、Ditto、CoACD 和物性辨识适配器可以独立替换，不会彼此冒充。

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
        "depth_unit": "native_u16_unscaled",
        "coordinate_frame": "camera_unregistered",
        "include_candidates": False,
    },
)
```

导入不修改源目录。文件在复制期间发生变化时整次导入失败；成功后每个流和附件都具有精确摘要。
这里不假定 uint16 原始值等于毫米，也不把未注册相机写成统一坐标系；尺度和 frame 必须由后续
独立度量证据与相机对齐产生。历史重建与仿真文件需另行审查后才能作为候选输入。

## 3. 安装并选择方法

外部算法通过工作区 `workers.d` 清单或 `physrealx.workers` 包入口注册独立进程/OCI
worker。先确认可执行状态，再绑定版本与配置：

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
| 单关节动力学观测 | `physrealx_mujoco_sysid_adapter` | 带来源、不确定度和留出验证的等效惯量、阻尼与摩擦 |

RDFC-GAN 无可直接复用的官方权重，不设为默认；ArtGen、PD²GS 等无法核验完整官方实现与许可证
的工作不注册为可执行方法。

## 4. 声明可执行配方

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

spec = RecipeExecutionSpec(
    resource_id="cabinet2.metric-camera",
    version="1.0.0",
    recipe=recipe,
    inputs=(
        OperationInput(name="capture", resource=source),
        OperationInput(name="metric_evidence", artifact=metric_alignment_evidence),
    ),
)
```

示例只展示“相机恢复 → 米制对齐”的严格绑定方式；`metric_alignment_evidence`
必须是先行发布的 `MetricAlignmentEvidence/1` 制品。完整配方用同样的 `CompletedStepOutput`
连接深度增强、几何重建、关节恢复、碰撞代理、物性辨识、完整资产组装、OpenUSD 打包和验收。上游没有产生真实制品时，
下游输入无法解析，配方也不会发布成功回执。每个输入绑定的 `contract` 还必须同时匹配
当前 worker 声明和上游制品契约，不能通过改名把单关节结果冒充为完整资产输入。
完整就绪链将米制几何、碰撞代理和资产组装列为独立阶段。不存在可执行 worker 或缺少版本化成功证据时均为
`BLOCKED`；`real/recon`、`sim/`或历史 ArtGS Mesh 只能作候选输入，不能直接满足这些门禁。

## 5. 构建与验收

```python
result = px.recipes.execute(spec)
for record in result.operations:
    print(record.step, record.operation.digest)
```

至少检查：相机重投影误差、深度有效率与边界误差、几何 Chamfer/F-score、关节轴/原点/范围误差、
碰撞近似误差、质量与摩擦拟合残差、穿模率、接触稳定性及多状态回放误差。没有真实参考值的指标
必须标记不可用，不能用渲染图代替定量验收。

每个成功阶段都发布 `OperationManifest`；终端阶段成功后才发布 `RecipeExecutionManifest`。
最终 `AssetSpec` 保存稳定 link/joint 树、分离的视觉/碰撞几何、单位化关节轴、合法限位以及带
`measured` / `inferred` / `default` 来源、不确定度和置信度的物理参数。任何修改都发布新版本。
