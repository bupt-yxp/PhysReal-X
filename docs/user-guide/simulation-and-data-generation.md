# 物理仿真与多模态数据生成

本模块在固定场景与任务上执行物理过程，重放同一状态轨迹生成多模态观测，并发布具有统一时间语义的 Episode。

## 完成目标

```text
SceneSpec + TaskSpec → 物理执行 → StateTrajectory → 多模态渲染
→ 传感器派生 → Episode 封装 → Episode 质量验收
```

核心仓库不附带生产级 MuJoCo、Isaac Sim、Genesis 或 BlenderProc 方法。实际运行必须由当前
项目环境已安装、固定版本并通过探针的方法完成。

## 1. 选择执行方法

| 需求 | 方法方向 |
| --- | --- |
| USD、RTX 与高保真铰接操作 | Isaac Sim / Isaac Lab 适配器 |
| 大规模并行训练 | Isaac Lab 或 Genesis 适配器 |
| 轻量刚体、关节和控制基准 | MuJoCo 适配器 |
| ROS 2 系统联调 | Gazebo Sim 适配器 |
| 离线高质量视觉模态 | BlenderProc 适配器 |
| 契约回归 | 内置 Reference backend |

后端必须声明刚体、关节、接触、批量执行、确定性 Reset、观测键、动作类型和设备。任务要求与后端能力不匹配时，应在创建会话前失败。

## 2. 执行物理轨迹

```python
simulation = px.simulations.run(
    scene_ref,
    task_ref,
    method="<installed-simulation-method>",
    config={"physics_dt": 0.002, "control_dt": 0.02, "seed": 2026},
)

state_trajectory = next(
    output.artifact for output in simulation.outputs if output.name == "state_trajectory"
)
```

运行身份必须固定场景、任务、后端、策略、种子、时间步和随机化实值。GPU 并行结果可以声明规格或统计复现；只有后端明确保证时才能声称位级确定。

## 3. 重放并生成多模态观测

```python
render = px.renders.generate(
    state_trajectory,
    method="<installed-render-method>",
)
```

渲染节点必须逐时间戳重放物理节点生成的 `StateTrajectory`，不能重新运行物理。否则 RGB、Depth、Mask、接触、关节状态和动作无法严格同步。

按需通过 `px.events.synthesize(...)` 从高帧率强度流合成 Event，通过 `px.pointclouds.project(...)` 从深度和相机标定反投影点云。每个派生模态都保留源帧、方法和参数引用。

## 4. 统一时间语义

平台分别记录物理、控制、渲染及各传感器时间步。每个普通步骤保存当前观测和由它产生的动作，末帧只保存终止观测：

```text
step 0: observation₀ + state₀ + action₀ + reward₀
step 1: observation₁ + state₁ + action₁ + reward₁
...
last:   observationₙ + stateₙ
```

原始层保留每个传感器的整数纳秒时间戳。插值、事件累积和时间窗口属于派生训练视图，不得改写原始时间线。

## 5. 封装并验证 Episode

```python
packed = px.episodes.pack(
    verified_run_evidence,
    method="<installed-episode-pack-method>",
)
```

`verified_run_evidence` 将物理轨迹、渲染流、任务判据和方法身份绑定为 `RunEvidence/1`，不是临时目录或单个轨迹文件。

`ExecutionLifecycle` 逐帧核对状态、渲染与 Episode 时间线。只有证据完全一致，才生成 `VerifiedEpisodeBundle`；多个 Bundle 可聚合为 `VerifiedEpisodeCollection`。

每条 Episode 至少记录：

- RGB、Depth、Mask、Event、点云与相机参数；
- 机器人和物体状态、动作与控制目标；
- 接触、力觉、奖励、成功和终止；
- 场景参数、随机化实值、方法版本和种子；
- 失败类别、安全事件和证据区间。

## 生成计划

生成任务应声明成功、可恢复失败、硬失败与 OOD 的目标比例，并设置 Episode 数、GPU 时间、存储空间、最低成功率和最大损坏率。达到成本或质量阈值时应保留已完成 Episode 与明确停止原因。

## 输出与验收

1. 信号名称、dtype、shape、单位和 frame 与任务一致；
2. 控制及传感器时间戳单调，跨模态残差低于预算；
3. 不含 NaN、Inf、越界动作、缺失制品或声明大小错误；
4. 成功、终止、失败原因与末帧语义一致；
5. 状态轨迹与全部渲染帧逐帧绑定；
6. 预算、停止原因和失败 Episode 完整保留。

仓库中的实际 cabinet2 训练实验生成了 96 个训练、24 个验证、24 个普通测试和 24 个 OOD MuJoCo Episode。该实验使用工程先验抽屉代理，只证明仿真与 Episode 代码实际执行，不代表真实物性资产已经通过仿真验证。

下一步进入[数据管理与模型训练](data-management-and-training.md)。
