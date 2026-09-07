# 领域模型

所有公共模型均为冻结对象：拒绝未知字段、NaN/Inf、非法标识和隐式单位。`model_copy(update=...)` 会重新执行完整校验。

## 基础引用

### `ArtifactRef`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `artifact_id` | `str` | 平台生成的不可变制品标识，格式为 `art_<uuid>` |
| `size_bytes` | `int` | 非负字节数 |
| `media_type` | `str` | IANA 媒体类型或平台扩展类型 |

### `VersionedRef`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `resource_type` | `ResourceType` | 资产、场景、任务、数据集、模型等 |
| `resource_id` | `str` | 稳定逻辑标识 |
| `version` | `str` | 显式语义版本 |
| `artifact_id` | `str | None` | 解析后必须存在，指向资源清单制品 |

`ResourceType` 包含 `observation_set`、`asset`、`scene`、`task`、`dataset`、`model`、
`evaluation`、`operation` 和 `recipe_execution`。

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
| 证据 | `evidence_state`、`evidence` |
| 扩展 | 有限的结构化 `metadata` |

`evidence_state` 为 `prior`、`identified` 或 `validated`。后两种状态必须引用不可变证据制品，
防止将默认值或纯视觉估计冒充实测物性。

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

## 观测集与配方执行

### `ObservationSetManifest`

| 字段 | 说明 |
| --- | --- |
| `resource_id` / `version` | 不可变观测集身份 |
| `importer` / `importer_version` | 解析源包的插件及版本 |
| `source_uri` | 源包位置，用于审计而非制品读取 |
| `streams` | 模态、状态、角色、编码、帧数、单位、frame、视图对齐策略和制品 |
| `attachments` | 标定、元数据、候选重建或候选仿真制品 |
| `issues` | 预检中保留的结构化警告 |

`ObservationStream.artifacts` 只允许一个打包制品，或与 `frame_count` 一致的逐帧制品。
RGB 流的 `view_alignment` 显式区分 `natural_features` 与 `fiducial_assisted`。
`FiducialSpec` 保存家族、静态 `layout_id` 以及可选的实测尺寸/布局 URI；非 RGB 流禁止声明该字段。

### `MethodRecipe`

```python
MethodRecipe(
    name: str,
    version: str,
    steps: tuple[RecipeStep, ...],
    outputs: tuple[RecipeFinalOutput, ...],
)
```

`RecipeStep` 将一个已注册的 `MethodSpec` 绑定到显式输入与输出契约。
`InitialRecipeInput` 引用初始不可变证据，`CompletedStepOutput` 只能引用已排在前面的阶段输出；
因此配方在构造时就拒绝前向引用、重复步骤名和未声明输入。

### `RecipeExecutionSpec` / `RecipeExecutionManifest`

`RecipeExecutionSpec` 固定执行身份、配方与初始 `OperationInput`。只有终端阶段成功后才发布
`RecipeExecutionManifest`，其中按顺序保存每个 `OperationManifest` 的精确引用及最终制品。
`MethodSpec` 包含完整 `MethodDescriptor`、可选权重制品和冻结配置。方法源码整体位于项目
`algorithms/`，通过可选依赖安装到当前项目环境，再由进程内适配器注册。

### `AssetValidationReport`

`profile` 可为 `schema`、`simulation`、`digital_twin` 或 `trainable`；任一 `error` 使
`passed=False`。`trainable` 要求已校验证据，并将默认物性与不完整关节限制升级为错误。

## 场景与任务模型

### `SceneSpec`

场景是独立版本化资源，不把某个资产隐式当作完整环境。核心字段包括：

| 字段 | 说明 |
| --- | --- |
| `resource_id` / `version` | 场景身份与语义版本 |
| `display_name` | 展示名称 |
| `entities` | 具有稳定名称、角色、资产精确引用和位姿的实体实例 |
| `metadata` | 有限的结构化扩展信息 |

`SceneEntitySpec.role` 为 `robot`、`object` 或 `environment`；实体名在场景内唯一。注册场景时，平台会把每个资产解析为带制品标识的精确引用。

### `TaskSpec`

包含场景精确引用、主体实体、语言指令、初始/成功/终止条件、观测模式、动作空间、最大步数和奖励折扣。`subject_entity` 必须对应场景中的稳定实体名。

### `Predicate`

```python
Predicate(
    signal: str,
    operator: Literal["ge", "gt", "le", "lt", "eq", "abs_le"],
    value: float,
    tolerance: float = 0.0,
)
```

## 方法执行证据

### `OperationSpec`

包含输出身份、完整 `MethodSpec`、命名的已解析资源/制品输入、结构化调用参数，以及每个输出的
精确名称与语义契约。
方法环境/算法配置属于 `MethodSpec.config`，“本次要生成什么”属于 `OperationSpec.arguments`，
两者不混用。

### `OperationManifest`

包含原 `OperationSpec`、按契约顺序返回的 `OperationOutput`、有限指标和含时区的起止时间。
输出必须已进入制品存储；输入引用、方法身份、输出名称、语义契约或顺序不一致均拒绝发布。

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

包含 Episode 记录、来源运行、观测/动作 Schema、总步数和结构化元数据。

## 学习与评测模型

### `TrainingSpec`

包含输出模型身份、数据集精确引用和 `policy.train` 能力的训练方法；配置与可选检查点位于
`TrainingSpec.trainer` 的 `MethodSpec` 中。

### `ModelManifest`

包含训练与推理 `MethodSpec`、数据集来源、权重制品、输入输出 Schema、规范化训练配置和指标。

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
record_id = canonical_record_id(spec)
```

规范化 JSON 使用 UTF-8、稳定键顺序且拒绝 NaN/Inf。`record_id` 只用于领域记录的确定性标识，
不读取文件，也不承担文件完整性校验。
