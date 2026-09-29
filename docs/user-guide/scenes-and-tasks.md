# 03 · 场景与任务生成

**输入：** 场景类型、PartNet-Mobility 资产目录、自然语言任务。

**流程：** 资产选择 → 房间布局 → 机器人放置 → 任务编译 → 轨迹执行 → 归档。

<!-- demo-preview:scene_and_task_generation -->

## 场景与机械臂

Demo 批量生成 `home`（家居）、`tabletop`（桌面）、`shelf`（货架）和 `industrial`（工业）四类场景，从本地 PartNet-Mobility 分别选择所需资产，并导入 MuJoCo Menagerie 的官方 Franka Emika Panda。每类场景均生成独立任务、轨迹、验收报告和 Blender 动画。

`method="SceneSmith"` 扫描完整 PartNet-Mobility 库，按任务语义、关节功能、几何完整性和尺度生成多个候选；随后以支撑关系、碰撞、房间边界和交互可达性完成布局。`collision_constraints` 复核静态布局，MuJoCo/Isaac Sim 继续检查动态接触和任务执行。每个未通过候选都会保留失败阶段和原因。

## 自然语言任务

```python
task = px.tasks.compile(
    "使用机械臂关闭笔记本电脑屏幕",
    scene_ref,
    method="articulated_templates",
    config={
        "resource_id": "partnet.laptop.close",
        "entity": "laptop",
        "joint": "joint_1",
        "robot": "robot_arm",
        "max_steps": 200,
    },
)
```

四类场景使用不同环境、目标物体和自然语言任务：家居关闭笔记本屏幕、桌面关闭瓶盖、货架关闭储物柜门、工业区关闭保险柜门。任务绑定场景实体、关节、初态、目标和成功条件；机械臂逆运动学生成计划，`mujoco_joint_trajectory` 执行并分别检查跟踪误差、接触力和任务目标。未通过门控的执行仍保留真实轨迹与诊断报告，并进入失败样本库。

门板任务根据接触点的闭合切线自动确定机械臂朝向，布局为底座和门板扫掠区域保留作业通道；底座与任一场景资产重叠时直接拒绝场景。执行器从两组稳定阻抗参数中按同一物理门控自动选择通过项，不能通过放宽阈值把失败执行标为成功。

## 四类输出

| 资源库 | 内容 |
| --- | --- |
| 场景库 | 房间、资产、机械臂及其位姿 |
| 任务脚本库 | 语言、观测、动作、初态与成功判据 |
| 操作轨迹库 | 机械臂命令与实际执行轨迹 |
| 成功/失败样本库 | 按跟踪、接触与任务目标三项门控归档的样本引用 |

四类场景的索引写入 `output/results.json`；模块四通过显式 `primary_scenario` 读取家居场景、任务实例与轨迹进行批量执行。当前 Demo 使用 PartNet-Mobility 物体：模块二的实拍重建尚未通过碰撞验收、没有发布可用资产，因此不会被模块三静默替换使用；后续只有模块二输出 `PASSED` 的版本化资产才能进入场景资产库。

[场景与任务 API](../api/scenes-and-tasks.md) · [下一步：多模态仿真](simulation-and-data-generation.md)
