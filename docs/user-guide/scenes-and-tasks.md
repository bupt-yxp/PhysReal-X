# 场景与任务

场景描述“世界如何组合”，任务描述“机器人需要完成什么以及如何判定”。二者分离后，同一资产可以进入多个场景，同一任务模板可以在多个资产实例上实例化。

## 场景规格

`SceneSpec` 组织：

- 机器人、物体和环境资产引用；
- 每个实体的逻辑名称、位姿和命名空间；
- 传感器类型、安装位置、内外参和频率；
- 灯光、材质、背景与环境参数；
- 物理时间步、渲染时间步和控制时间步；
- 域随机化变量、分布、相关性和约束；
- 碰撞过滤、安全区和工作空间边界。

```python
scene = px.scenes.compose(
    scene_id="scene.drawer-kitchen",
    robot="franka.standard@1.0.0",
    assets={"drawer": "drawer.real-001@1.0.0"},
    template="kitchen.countertop@2.1.0",
    sensors=("camera.front", "wrist.rgbd", "wrist.force_torque"),
)
```

## 随机化不是字段堆积

每个随机变量都应声明作用范围、采样时机和分布：

| 时机 | 适合参数 |
| --- | --- |
| `build` | 资产实例、布局拓扑、传感器数量 |
| `episode_reset` | 初始位姿、光照、材质、摩擦、质量扰动 |
| `step` | 传感噪声、时延、外力；谨慎使用 |

相关变量应联合采样。例如静摩擦通常不应低于动摩擦；相机内参扰动和图像畸变参数需要保持物理可实现。

## 从语言到可执行任务

自然语言是任务输入，不是最终任务合同：

```text
“把杯子放进抽屉”
  ↓
识别目标实体与可供性
  ↓
生成任务图：打开 → 抓取 → 移动 → 放置 → 释放
  ↓
解析观测与动作模式
  ↓
编译初始、成功、终止和安全条件
  ↓
静态验证与可达性预检
```

```python
task = px.tasks.compile(
    instruction="把杯子放进抽屉",
    scene=scene,
    task_id="task.place-cup-in-drawer",
    robot="robot",
    targets={"container": "drawer", "object": "cup"},
)
```

编译器可以调用规则、任务模板、LLM 规划器和运动规划器，但最终必须输出确定、可验证的 `TaskSpec`。

## 初始条件

初始条件用于验证 reset 后的世界，而不是描述随机化范围：

```yaml
initial_conditions:
  - {signal: drawer.joint_position, operator: le, value: 0.05, unit: m}
  - {signal: cup.in_gripper, operator: eq, value: false}
```

若后端 reset 后不满足条件，Episode 应失败为环境初始化错误，而不是进入策略执行。

## 成功条件与终止条件

成功条件必须可由声明信号或事件判定：

```yaml
success_conditions:
  all:
    - {signal: cup.inside.drawer, operator: eq, value: true}
    - {signal: cup.linear_speed, operator: abs_le, value: 0.02, unit: m/s}
    - {hold_for_steps: 10}
```

终止条件包括：

- 达到任务成功；
- 进入不可恢复失败；
- 触发安全事件；
- 达到最大步数或时间预算；
- 后端或控制器报告异常。

“看起来完成”不够。放置类任务通常还需要稳定保持、速度阈值和接触关系条件。

## 观测模式

每个信号声明：名称、dtype、shape、unit、frame、frequency 与同步容差。

```yaml
observations:
  - name: camera.front.rgb
    dtype: uint8
    shape: [480, 640, 3]
    unit: "1"
    frame: camera.front
    frequency_hz: 30
  - name: wrist.force
    dtype: float32
    shape: [3]
    unit: N
    frame: wrist
    frequency_hz: 200
```

高频力觉和低频图像保留各自传感器时间戳，不应伪造成完全同频数据。

## 动作空间

常用动作类型：

- 关节位置、速度或力矩；
- 末端绝对/增量位姿；
- 夹爪开合；
- 离散技能或任务图节点；
- 多执行器复合动作。

动作必须声明控制 frame、范围、频率、归一化和安全裁剪策略。适配器负责把平台动作转换为具体控制器命令。

## 任务变体与评测切分

训练与评测不应只改变随机种子。场景矩阵应显式控制：

- 见过 / 未见物体实例；
- 见过 / 未见场景与布局；
- 关节、摩擦、质量和传感噪声范围；
- 语言表达变化；
- 长尾遮挡、接触和恢复条件。

同一矩阵版本一旦用于公开结果，不应原地修改。
