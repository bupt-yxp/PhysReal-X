# 场景与任务 API

场景明确组合机器人、物体和环境资产；任务绑定场景实体并声明观测、动作、初始条件、成功条件与
终止条件。自然语言和生成器只能产生待校验草案，不能绕过公共模型直接执行。

## 场景

### `scenes.register`

```python
px.scenes.register(spec: SceneSpec) -> VersionedRef
```

平台解析每个实体资产为带摘要引用，验证实体名唯一后原子发布场景。任务逻辑不得依赖实体数组下标。

### `scenes.get` / `scenes.list`

```python
px.scenes.get(ref: VersionedRef) -> SceneSpec
px.scenes.list() -> list[VersionedRef]
```

```python
from physrealx import SceneEntityRole, SceneEntitySpec, SceneSpec

scene = px.scenes.register(
    SceneSpec(
        resource_id="drawer_lab",
        version="1.0.0",
        display_name="Drawer lab",
        entities=(
            SceneEntitySpec(
                name="drawer",
                role=SceneEntityRole.OBJECT,
                asset=drawer_ref,
            ),
        ),
    )
)
```

## 任务

### `tasks.register`

```python
px.tasks.register(spec: TaskSpec) -> VersionedRef
```

注册时解析场景摘要，并确认 `subject_entity` 存在。谓词只能引用已声明的标量数值观测，动作和观测
信号名称不能冲突。

### `tasks.get` / `tasks.list`

```python
px.tasks.get(ref: VersionedRef) -> TaskSpec
px.tasks.list() -> list[VersionedRef]
```

`TaskSpec` 至少固定：

- 场景引用和主体实体；
- 人类可读指令；
- 初始、成功与终止谓词；
- 观测 `SignalSpec` 和带上下界的 `ActionSpec`；
- 最大步数与折扣。

## 生成、编译与轨迹

```python
draft = px.scenes.generate(
    {"domain": "kitchen", "seed": 2026, "asset_ids": ["cabinet2"]},
    method="robocasa365",
)
scene_result = px.scenes.compose(
    draft.outputs[0].artifact,
    method="constraint_solver",
)
task_result = px.tasks.compile(
    "打开柜门并把杯子放入柜体",
    scene_ref,
    method="bddl",
)
trajectory = px.trajectories.generate(
    task_result.outputs[0].artifact,
    method="curobo",
)
validation = px.tasks.validate(task_ref, method="task_probe")
```

| API | 能力 | 强制输出 |
| --- | --- | --- |
| `scenes.generate` | `scene.generate` | `scene_draft` |
| `scenes.compose` | `scene.compose` | `scene_spec` + `constraint_report` |
| `tasks.compile` | `task.compile` | `task_spec` |
| `trajectories.generate` | `trajectory.generate` | `trajectory` + `planning_report` |
| `tasks.validate` | `task.validate` | `validation_report` |

生成结果是待发布制品；只有将其反序列化为公共 `SceneSpec` / `TaskSpec` 并通过
`register` 后，才能被运行引用。LLM 只能生成结构化草案；平台仍会执行实体解析、Schema、
可满足性、可达性、碰撞和冻结种子仿真检查。生成的 Python 代码不允许直接执行。

## 主要错误

- `ResourceNotFoundError`：场景引用的资产或任务引用的场景不存在；
- `ResourceConflictError`：引用摘要或不可变资源身份冲突；
- `DomainValidationError`：实体、信号、谓词、动作边界或任务绑定不合法。
