# 场景与任务 API

## Scene API

### `scenes.register`

```python
px.scenes.register(
    spec: SceneSpec,
    *,
    idempotency_key: str | None = None,
) -> ResolvedRef[SceneSpec]
```

解析全部资产引用、验证实体命名、位姿、传感器和随机化约束后发布场景。

### `scenes.compose`

```python
px.scenes.compose(
    *,
    resource_id: str,
    version: str = "1.0.0",
    template: SceneLike | None = None,
    robot: AssetLike,
    assets: Mapping[str, AssetLike],
    sensors: Sequence[SensorSpec | str] = (),
    randomization: RandomizationSpec | None = None,
) -> ResolvedRef[SceneSpec]
```

从模板和资产引用构建场景并直接注册。`assets` 的键成为场景内稳定实体名。

### `scenes.get`

```python
px.scenes.get(ref: SceneLike) -> SceneSpec
```

### `scenes.validate`

```python
px.scenes.validate(
    scene: SceneSpec | SceneLike,
    *,
    backend: str | None = None,
    check_reachability: bool = False,
) -> ValidationReport
```

### `scenes.materialize`

```python
px.scenes.materialize(
    scene: SceneLike,
    *,
    backend: str,
    seed: int,
) -> JobRef[SceneInstance]
```

用于调试场景转换和随机化采样。`SceneInstance` 是有生命周期的后端实例，不是版本化公共资源；使用完必须释放。

## Task API

### `tasks.register`

```python
px.tasks.register(
    spec: TaskSpec,
    *,
    idempotency_key: str | None = None,
) -> ResolvedRef[TaskSpec]
```

### `tasks.compile`

```python
px.tasks.compile(
    draft: TaskDraft | str,
    *,
    scene: SceneLike,
    resource_id: str | None = None,
    version: str = "1.0.0",
    compiler: str = "default",
    config: Mapping[str, JSONValue] | None = None,
) -> ResolvedRef[TaskSpec]
```

编译自然语言或结构化草案，输出确定性任务规格。若编译结果存在歧义或无法构造可判定成功条件，抛出 `TaskCompilationError`，不自动猜测。

### `tasks.get`

```python
px.tasks.get(ref: TaskLike) -> TaskSpec
```

### `tasks.validate`

```python
px.tasks.validate(
    task: TaskSpec | TaskLike,
    *,
    scene: SceneLike | None = None,
    backend: str | None = None,
    policy: str | None = None,
) -> ValidationReport
```

检查信号、条件、动作空间、实体引用、可达性以及后端/策略能力。

### `tasks.instantiate`

```python
px.tasks.instantiate(
    template: TaskLike,
    *,
    scene: SceneLike,
    bindings: Mapping[str, str],
    version: str,
) -> ResolvedRef[TaskSpec]
```

将参数化任务模板绑定到具体场景实体。

## 示例

```python
scene = px.scenes.compose(
    resource_id="scene.drawer-table",
    robot="franka.standard@1.0.0",
    assets={"drawer": "drawer.real-001@1.0.0"},
    sensors=("camera.front", "wrist.force_torque"),
)

task = px.tasks.compile(
    "打开抽屉至少 30 厘米",
    scene=scene,
    resource_id="task.open-drawer",
)

px.tasks.validate(task, scene=scene, backend="isaac_lab").raise_for_errors()
```

## 主要错误

- `SceneCompositionError`：实体、位姿或模板绑定非法；
- `RandomizationConstraintError`：随机变量违反联合约束；
- `TaskCompilationError`：草案无法编译为可执行规格；
- `SignalSchemaError`：观测、动作或条件信号不一致；
- `CapabilityMismatchError`：后端或策略不支持任务要求。
