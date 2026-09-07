# 场景与任务 API

场景明确组合机器人、物体和环境资产；任务绑定场景实体并声明观测、动作、初始条件、成功条件与
终止条件。自然语言和生成器只能产生待校验草案，不能绕过公共模型直接执行。
当前内置有界布局、开关关节模板和经过 MuJoCo 执行检查的轨迹方法。

## 场景

### `scenes.register`

```python
px.scenes.register(spec: SceneSpec) -> VersionedRef
```

平台解析每个实体资产为带制品标识的版本引用，验证实体名唯一后原子发布场景。任务逻辑不得依赖实体数组下标。

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

注册时解析场景版本，并确认 `subject_entity` 存在。谓词只能引用已声明的标量数值观测，动作和观测
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

| API | 已实现方法 | 输出 |
| --- | --- | --- |
| `scenes.generate` | `bounded_layout` | `scene_draft` |
| `scenes.compose` | `collision_constraints` | `scene_spec` / `constraint_report` |
| `tasks.compile` | `articulated_templates` | `task_spec` |
| `tasks.validate` | `joint_constraints` | `validation_report` |
| `trajectories.generate` | `mujoco_joint_trajectory` | `trajectory` / `planning_report` |

请求与范围见[场景任务指南](../user-guide/scenes-and-tasks.md)。布局固定种子、几何边界和最大尝试次数；模板只支持明确开关意图，绑定具体实体/关节；轨迹通过真实 MuJoCo 检查初态、接触、跟踪与成功判据。

生成结果经公共 `SceneSpec` / `TaskSpec` 校验与 `register` 后用于运行；运动验证不能由静态布局通过替代。

## 主要错误

- `ResourceNotFoundError`：场景引用的资产或任务引用的场景不存在；
- `ResourceConflictError`：制品引用或不可变资源身份冲突；
- `DomainValidationError`：实体、信号、谓词、动作边界或任务绑定不合法。
