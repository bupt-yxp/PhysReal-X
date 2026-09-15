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

`scenes.place_linear_actuator(scene, entity=..., joint=..., actuator=..., name=..., initial_position=..., clearance_m=..., resource_id=...)` 根据移动关节与碰撞几何生成执行器位姿，返回待注册的 `SceneSpec`。目标支持平移或具有水平开合位移的旋转关节；旋转位置使用 rad，平移位置使用 m。返回场景的 `metadata.actuator_placement.closing_travel_m` 提供推杆闭合行程，并校验执行器行程范围。接触起点与终点按推杆实际接触窗口裁剪碰撞表面计算，窗口外把手等凸起不影响行程。执行器沿自身局部 -X 方向推进；初始位置、正间隙、实体名称及引用由项目校验。该几何放置不替代后续动力学接触验证。

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

`CameraSpec.look_at(position, target, ...)` 生成 OpenGL 相机位姿并拒绝退化视线；`CameraSpec.signals(name, frequency_hz=...)` 按真实分辨率生成同步信号契约。`px.tasks.with_sensors(task, scene_ref, cameras, resource_id=..., frequency_hz=...)` 进一步从场景资产计算部件数量、生成相机与刚体信号，保留任务动作和成功条件，返回待注册任务。重复信号和场景引用不一致会报错。

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
