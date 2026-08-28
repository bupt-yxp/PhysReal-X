# 领域模型

所有公共模型均为冻结对象：拒绝未知字段、NaN/Inf、非法标识和隐式单位。`model_copy(update=...)` 会重新执行完整校验。

## 基础引用

### `ArtifactRef`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `digest` | `str` | `sha256:` 开头的内容摘要 |
| `size_bytes` | `int` | 非负字节数 |
| `media_type` | `str` | IANA 媒体类型或平台扩展类型 |

### `VersionedRef[T]`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `resource_type` | `ResourceType` | 资产、场景、任务、数据集、模型等 |
| `resource_id` | `str` | 稳定逻辑标识 |
| `version` | `str` | 显式语义版本 |
| `digest` | `str | None` | 解析后必须存在 |

## 空间与信号

### `PoseSpec`

```python
PoseSpec(
    translation_m: tuple[float, float, float] = (0, 0, 0),
    rotation_xyzw: tuple[float, float, float, float] = (0, 0, 0, 1),
)
```

四元数必须归一化，旋转顺序固定为 `xyzw`。

### `SignalSpec`

| 字段 | 说明 |
| --- | --- |
| `name` | 全局或任务内唯一信号名 |
| `dtype` / `shape` | 机器可验证的数据模式 |
| `unit` | SI 或明确机器符号，标量无单位使用 `1` |
| `frame` | 逻辑坐标系；图像信号使用相机 frame |
| `frequency_hz` | 期望采样频率 |

### `DataValue`

精确包含一个载荷：小型 `inline` 值或大对象 `artifact`。同时携带 dtype、shape、unit、frame 和可选传感器时间戳。

## 资产模型

### `AssetSpec`

| 字段组 | 主要类型 |
| --- | --- |
| 身份 | `resource_id`, `version`, `display_name`, `category`, `tags` |
| 坐标 | `FrameSpec`, `root_pose` |
| 结构 | `tuple[LinkSpec, ...]`, `tuple[JointSpec, ...]` |
| 表示 | `tuple[GeometrySpec, ...]`, `MaterialSpec`, `CollisionSpec` |
| 物理 | `tuple[BodyPhysicsSpec, ...]` |
| 来源与验证 | `ProvenanceSpec`, `CalibrationRef`, `ValidationReportRef` |

### `JointSpec`

支持 `fixed`、`revolute`、`prismatic`，包含 parent/child、局部关节位姿、单位轴、限位、阻尼、摩擦、速度与力限制。

### `ScalarParameter`

```python
ScalarParameter(
    value: float,
    unit: str,
    uncertainty: float,
    source: Literal["measured", "inferred", "default"],
    confidence: float,
)
```

## 场景与任务模型

### `SceneSpec`

场景是独立版本化资源，不把某个资产隐式当作完整环境。核心字段包括：

| 字段 | 说明 |
| --- | --- |
| `resource_id` / `version` | 场景身份与语义版本 |
| `display_name` | 展示名称 |
| `entities` | 具有稳定名称、角色、资产精确引用和位姿的实体实例 |
| `metadata` | 有限的结构化扩展信息 |

`SceneEntitySpec.role` 为 `robot`、`object` 或 `environment`；实体名在场景内唯一。注册场景时，平台会把每个资产解析为带摘要的精确引用。

### `RandomVariableSpec`

声明路径、分布、参数、采样时机、相关组、裁剪和物理约束。

### `TaskSpec`

包含场景精确引用、主体实体、语言指令、初始/成功/终止条件、观测模式、动作空间、最大步数和奖励折扣。`subject_entity` 必须对应场景中的稳定实体名。

### `Predicate`

```python
Predicate(
    signal: str,
    operator: Literal["ge", "gt", "le", "lt", "eq", "abs_le"],
    value: float | bool,
    unit: str = "1",
    tolerance: float = 0.0,
)
```

## 运行与 Episode

### `RunSpec`

包含场景、任务、后端、策略、Episode 数、根种子、并行度、记录信号、输出数据集、质量门、预算和标签。

### `EpisodeStep`

| 字段 | 语义 |
| --- | --- |
| `index` | 从零连续递增 |
| `timestamp_seconds` | 控制时间线 |
| `observations` / `state` | 当前观测和状态 |
| `action` / `reward` / `discount` | 从当前观测执行的动作与结果；末帧为空 |
| `is_first` / `is_last` / `is_terminal` | 步骤边界语义 |
| `events` | 结构化事件代码 |

### `Episode`

包含来源、种子、场景/任务精确引用、后端/策略版本、步骤、成功状态、失败原因和运行元数据。Episode 不再重复保存单资产引用，全部资产均由场景清单确定。

### `DatasetManifest`

包含 Episode 摘要、来源运行、信号模式、split、统计、质量报告、数据卡和使用限制。

## 学习与评测模型

### `TrainingSpec`

包含训练器、数据集或环境、split、种子、配置、资源、validation、检查点策略和输出模型身份。

### `ModelManifest`

包含训练器、数据集与环境来源、权重、输入输出模式、训练配置、指标、检查点和推理适配器。

### `EvaluationSpec` / `EvaluationReport`

规格固定模型、矩阵、后端、种子、重复数和指标；报告保存聚合指标、分层结果、置信区间、失败案例和 Episode 数据集。

## 部署与反馈模型

### `DeploymentRecord`

保存模型、机器人、任务、安全配置、控制器、运行模式、状态、事件和真实 Episode 引用。

### `FailureCase`

```python
FailureCase(
    episode_id: str,
    category: FailureCategory,
    stage: str,
    message: str,
    evidence: tuple[EvidenceRef, ...],
    confidence: float,
)
```

### `RegenerationPlan`

保存目标失败簇、待调整变量、采样分布、数据比例、预算、停止条件、质量门和输出数据集身份。

## 序列化

```python
spec = TaskSpec.from_file("task.yaml")
spec.to_file("task.normalized.json")

payload = spec.model_dump(mode="json")
digest = spec.digest()
```

规范化 JSON 使用 UTF-8、稳定键顺序、无 NaN/Inf；内容摘要不依赖空格和输入文件格式。
