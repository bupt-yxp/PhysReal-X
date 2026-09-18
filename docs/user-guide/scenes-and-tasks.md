# 场景与任务生成

模块三根据场景类型从 PartNet-Mobility 选择资产，通过 `physcene_guidance → collision_constraints → articulated_templates → joint_constraints → mujoco_joint_trajectory` 生成并实际验证任务。

## 布局

`assets.select_partnet_mobility` 根据 home/tabletop/shelf/industrial 场景类型和随机种子，从本地资产目录确定性选择并注册所需物体。`scenes.generate` 声明空间关系和布局边界，布局采用资产碰撞几何的保守包围盒。

`objects` 每项给出名称、资产引用与角色，`relations` 声明承载和前后左右关系。`scenes.compose(..., method="collision_constraints")` 复核静态重叠，运动碰撞由 MuJoCo 检查。

## 语言模板与控制

```python
operation = px.tasks.compile(
    "使用机械臂关闭笔记本电脑屏幕",
    scene_ref,
    method="articulated_templates",
    config={
        "resource_id": "partnet.laptop.close",
        "entity": "laptop",
        "joint": "joint_1",
        "robot": "robot_arm",
        "max_steps": 100,
    },
)
```

模板仅解释明确的开/关意图，再绑定实际实体和关节。歧义、否定、缺失关节和未声明作用力上限会被拒绝。输出 `TaskSpec` 固定初态、目标、单位、动作上下限及成功判据；不把字符串模板当作通用语言规划。

`tasks.validate(..., method="joint_constraints")` 检查控制关节属于场景资产及命令边界。控制执行器与任务作用物体可以是不同实体。

接触任务通过 `robot="robot_arm"` 使用场景中经逆运动学求解的多关节计划，并用
`initial_fraction` 声明被动物体初态。模板检查机器人角色、关节、位置向量和作用力上限；
`joint.position/velocity/effort` 描述机器人关节，成功谓词引用物体的具名关节观测。
模块三 Notebook 使用 Blender 展示房间布局和 Franka Panda 的执行结果，并列出本次生成的场景、任务脚本、操作轨迹与成功/失败样本索引。批量失败样本由模块四生成。

## 轨迹

`trajectories.generate(..., method="mujoco_joint_trajectory")` 消费 `TaskInstance`，其中包含任务引用、`MujocoConfig` 与目标向量。采用五次平滑位置轨迹，逐步执行 MuJoCo，检查初态、跟踪误差、接触力及最终成功判据；缺少位置/接触观测、越界或超过任务时域时拒绝发布。

模块四进一步执行场景任务、随机化及成功/失败样本扩展。更复杂机器人运动规划可以接入同一协议，但目前没有把未接入的 RoboCasa、BDDL 或 cuRobo 注册为可用方法。
