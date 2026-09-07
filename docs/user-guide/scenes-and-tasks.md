# 场景与任务生成

模块三读取已登记物理资产，通过 `bounded_layout → collision_constraints → articulated_templates → joint_constraints → mujoco_joint_trajectory` 生成并实际验证任务。

## 布局

`scenes.generate` 请求显式声明 `resource_id`、`version`、`display_name`、`domain`、`objects`、`xy_bounds_m` 和 `seed`。场景类型支持 home/tabletop/shelf/industrial；这决定任务场景标签，布局采用真实碰撞几何的保守包围盒。

`objects` 每项给出名称、资产引用与角色。采样受空间、间距、朝向、支撑高度和尝试次数约束；不可行时报告失败。`scenes.compose(..., method="collision_constraints")` 复核静态重叠，运动碰撞另由真实引擎检查。当前资产根节点固定；布局不是可达性或抓取规划的证明。

## 语言模板与控制

```python
operation = px.tasks.compile(
    "打开上方抽屉",
    scene_ref,
    method="articulated_templates",
    config={
        "resource_id": "cabinet.open",
        "entity": "cabinet",
        "joint": "gui_drawer_1.joint",
        "target_fraction": 0.85,
        "max_steps": 100,
    },
)
```

模板仅解释明确的开/关意图，再绑定实际实体和关节。歧义、否定、缺失关节和未声明作用力上限会被拒绝。输出 `TaskSpec` 固定初态、目标、单位、动作上下限及成功判据；不把字符串模板当作通用语言规划。

`tasks.validate(..., method="joint_constraints")` 检查控制关节属于场景资产及命令边界。控制执行器与任务作用物体可以是不同实体。

接触任务在模板配置中增加 `actuation={"entity": "pusher", "joint": "extension", "initial_position": 0.0, "target_position": 0.195}`，
并用 `initial_fraction` 声明被动物体初态。模板检查驱动实体的机器人角色、关节与作用力上限；
`joint.position/velocity/effort` 描述驱动关节，成功谓词引用物体的具名关节观测。
模块三 Notebook 给出推杆关闭下层抽屉的完整配置，避免把机器人命令完成误判为物体任务完成。

## 轨迹

`trajectories.generate(..., method="mujoco_joint_trajectory")` 消费 `TaskInstance`，其中包含任务引用、`MujocoConfig` 与目标向量。采用五次平滑位置轨迹，逐步执行 MuJoCo，检查初态、跟踪误差、接触力及最终成功判据；缺少位置/接触观测、越界或超过任务时域时拒绝发布。

模块四进一步执行直线推杆与被动抽屉接触、随机化及成功/失败样本扩展。更复杂机器人运动规划可以接入同一协议，但目前没有把未接入的 RoboCasa、BDDL 或 cuRobo 注册为可用方法。
