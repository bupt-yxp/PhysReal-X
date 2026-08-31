# 场景与任务生成

本模块把已验证资产组合成环境，并将用户目标编译为可执行、可判定的任务。场景描述“世界如何组成”，任务描述“机器人需要完成什么”。

## 完成目标

```text
选择资产 → 生成候选布局 → 约束求解 → 编译任务谓词 → 轨迹预检 → 任务验收
```

输出为不可变 `SceneSpec` 和 `TaskSpec`。核心仓库尚未内嵌 RoboCasa、BDDL、cuRobo 等方法；
示例方法必须先作为项目可选依赖安装、注册并通过探针。

## 1. 生成并组合场景

```python
draft = px.scenes.generate(
    {
        "domain": "kitchen",
        "seed": 2026,
        "assets": ["drawer.real-001@1.0.0"],
        "robot": "franka.standard@1.0.0",
    },
    method="robocasa365",
)

composed = px.scenes.compose(
    draft.outputs[0].artifact,
    method="constraint_solver",
)
```

候选场景必须经过碰撞、支撑、可达性、视野和工作空间约束求解，再将通过公共模型校验的 `SceneSpec` 交给 `px.scenes.register(...)` 发布。

场景至少声明实体引用与位姿、传感器及频率、物理/控制/渲染步长、碰撞过滤、安全区和随机化参数。每个随机变量应写明作用范围、分布和采样时机，相关物理量应联合约束。

## 2. 编译任务

```python
compiled = px.tasks.compile(
    instruction="把杯子放进抽屉",
    scene=scene_ref,
    method="bddl",
    config={"robot": "robot", "container": "drawer", "object": "cup"},
)

task_spec = next(output.artifact for output in compiled.outputs if output.name == "task_spec")
```

自然语言只能生成任务草案。最终 `TaskSpec` 必须显式绑定场景实体，并声明观测、动作、初始条件、成功条件、终止条件和安全限制。LLM 不得直接生成并执行 Python。

## 3. 定义可判定条件

```yaml
initial_conditions:
  - {signal: drawer.joint_position, operator: le, value: 0.05, unit: m}
  - {signal: cup.in_gripper, operator: eq, value: false}

success_conditions:
  all:
    - {signal: cup.inside.drawer, operator: eq, value: true}
    - {signal: cup.linear_speed, operator: abs_le, value: 0.02, unit: m/s}
    - {hold_for_steps: 10}
```

Reset 后不满足初始条件属于环境初始化错误。“看起来完成”不是成功判据；放置、接触和操作任务通常还需要稳定保持、速度阈值和接触关系。

## 4. 声明观测与动作

每个观测信号必须给出名称、dtype、shape、单位、frame、频率和同步容差。高频力觉与低频图像保留各自时间戳，不伪造成同频数据。

动作空间可以是关节位置/速度/力矩、末端位姿、夹爪、离散技能或复合动作，但必须固定控制 frame、范围、频率、归一化和安全裁剪策略。

## 5. 生成轨迹并验证

```python
trajectory = px.trajectories.generate(
    validated_task_instance,
    method="curobo",
)

validation = px.tasks.validate(
    task_spec,
    method="task_validator",
)
```

轨迹规划用于验证几何可达性和生成专家候选，不改变任务成功条件。任务验收应覆盖 Schema、对象绑定、谓词可满足性、规划可达性、碰撞和安全约束。

## 训练与评测切分

任务矩阵应显式区分见过/未见资产、布局、语言表达和物理参数范围，并覆盖遮挡、接触和恢复长尾。评测矩阵发布后不得原地修改；只改变随机种子不足以构成独立评测。

## 输出与验收

- 所有资产和机器人引用均解析为精确版本；
- Reset 后初始谓词可满足；
- 成功、终止和安全条件均由声明信号确定性计算；
- 观测、动作、单位、frame 与目标后端能力一致；
- 训练、验证和测试矩阵没有资产或采集批次泄漏。

下一步进入[物理仿真与多模态数据生成](simulation-and-data-generation.md)。
