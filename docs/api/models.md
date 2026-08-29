# 领域模型

所有公共模型均为冻结对象：拒绝未知字段、NaN/Inf、非法标识和隐式单位。`model_copy(update=...)` 会重新执行完整校验。

## 基础引用

### `ArtifactRef`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `digest` | `str` | `sha256:` 开头的内容摘要 |
| `size_bytes` | `int` | 非负字节数 |
| `media_type` | `str` | IANA 媒体类型或平台扩展类型 |

### `VersionedRef`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `resource_type` | `ResourceType` | 资产、场景、任务、数据集、模型等 |
| `resource_id` | `str` | 稳定逻辑标识 |
| `version` | `str` | 显式语义版本 |
| `digest` | `str | None` | 解析后必须存在 |

`ResourceType` 包含 `observation_set`、`asset_build`、`asset`、`scene`、`task`、`dataset`、`model` 和 `evaluation`。

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
| 身份 | `resource_id`, `version`, `display_name` |
| 坐标 | `coordinate_frame`, `up_axis`, `handedness`, `root_pose` |
| 结构 | `root_link`, `links`, `joints` |
| 表示 | `geometries`，每项明确 `visual` 或 `collision` 角色 |
| 物理 | `tuple[BodyPhysicsSpec, ...]` |
| 扩展 | 有限的结构化 `metadata` |

### `JointSpec`

支持 `fixed`、`revolute`、`prismatic`，包含 parent/child、局部关节位姿、单位轴、限位、阻尼、摩擦、速度与力限制。

### `ScalarParameter`

```python
ScalarParameter(
    value: float,
    uncertainty: float,
    source: Literal["measured", "inferred", "default"],
    confidence: float,
)
```

物理量单位由字段名固定，例如 `mass_kg` 和 `inertia_kg_m2`，不接受隐式或自由文本单位。

## 观测集与数字孪生

### `ObservationSetManifest`

| 字段 | 说明 |
| --- | --- |
| `resource_id` / `version` | 不可变观测集身份 |
| `importer` / `importer_version` | 解析源包的插件及版本 |
| `source_uri` | 源包位置，用于审计而非制品读取 |
| `streams` | 模态、状态、角色、编码、帧数、单位、frame 和制品 |
| `attachments` | 标定、元数据、候选重建或候选仿真制品 |
| `issues` | 预检中保留的结构化警告 |

`ObservationStream.artifacts` 只允许一个打包制品，或与 `frame_count` 一致的逐帧制品。

### `DigitalTwinRecipe`

```python
DigitalTwinRecipe(
    name: str,
    version: str,
    pipeline: str,
    pipeline_version: str,
    steps: tuple[DigitalTwinStep, ...],
)
```

`DigitalTwinStep` 将有序 `DigitalTwinStage` 绑定到能力匹配的 `MethodSpec`。`MethodSpec`
包含完整 `MethodDescriptor`、可选权重 `ArtifactRef` 和冻结配置；描述符固定源码 revision、
许可证、运行环境以及输入输出契约。

### `DigitalTwinBuildSpec` / `DigitalTwinBuildManifest`

`DigitalTwinBuildSpec` 需要带摘要的 `observation_set` 引用、目标资产身份和完整配方；发布后的 manifest 增加逐阶段方法、时间、指标、制品、最终资产和验证报告。

### `AssetValidationReport`

`profile` 可为 `schema`、`simulation` 或 `digital_twin`；任一 `error` 使 `passed=False`，`warning` 不阻止发布但会进入构建清单。

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

包含场景/任务引用、后端与策略 `MethodSpec`、Episode 数、根种子、记录信号、输出数据集身份和标签。
后端必须声明 `simulation.run`，策略必须声明 `policy.inference`。

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

包含来源、种子、场景/任务精确引用、后端/策略完整方法证据、步骤、成功状态、失败原因和运行元数据。
Episode 不重复保存单资产引用，全部资产由场景清单确定。

### `DatasetManifest`

包含 Episode 摘要、来源运行、观测/动作 Schema、总步数和结构化元数据。

## 学习与评测模型

### `TrainingSpec`

包含输出模型身份、数据集精确引用和 `policy.train` 能力的训练方法；配置与可选检查点位于
`TrainingSpec.trainer` 的 `MethodSpec` 中。

### `ModelManifest`

包含训练与推理 `MethodSpec`、数据集来源、权重摘要、输入输出 Schema、规范化训练配置和指标。

### `EvaluationReport`

报告保存模型、评测数据集、成功率、结构化指标和失败案例，并以不可变资源版本发布。

### `FailureCase`

```python
FailureCase(
    episode_id: str,
    category: FailureCategory,
    message: str,
)
```

## 序列化

```python
spec = TaskSpec.model_validate(task_payload)
payload = spec.model_dump(mode="json")
digest = sha256_digest(spec)
```

规范化 JSON 使用 UTF-8、稳定键顺序、无 NaN/Inf；内容摘要不依赖空格和输入文件格式。
