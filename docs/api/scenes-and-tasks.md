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

## 生成与编译方法

场景生成、任务编译和轨迹规划通过方法能力注册，不向 `SceneSpec` / `TaskSpec` 添加算法专属字段：

```python
px.methods.list(MethodCapability.SCENE_GENERATE)
px.methods.list(MethodCapability.TASK_COMPILE)
px.methods.list(MethodCapability.TASK_PLAN)
px.methods.list(MethodCapability.TASK_VALIDATE)
```

生成结果必须先转换为公共模型，再经过实体解析、Schema、约束、可达性、碰撞和冻结种子仿真检查。
模型生成的 Python 代码不能直接进入工作节点。

## 主要错误

- `ResourceNotFoundError`：场景引用的资产或任务引用的场景不存在；
- `ResourceConflictError`：引用摘要或不可变资源身份冲突；
- `DomainValidationError`：实体、信号、谓词、动作边界或任务绑定不合法。
