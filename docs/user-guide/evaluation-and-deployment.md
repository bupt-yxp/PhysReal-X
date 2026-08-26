# 评测、部署与反馈闭环

平台的最终价值不是在仿真中获得高成功率，而是让策略在可解释的评测体系下进入真实机器人，并把失败转化为下一轮数据生成计划。

## 评测矩阵

评测矩阵是冻结的场景与任务集合：

```yaml
resource_id: evaluation.drawer-heldout
version: 1.0.0
axes:
  asset_instance: [seen, unseen]
  scene_layout: [seen, unseen]
  lighting: [nominal, hard]
  friction: [low, nominal, high]
  language: [canonical, paraphrase]
repetitions: 20
```

矩阵保证不同模型在相同条件下比较，并防止只报告有利场景。

## 提交评测

```python
evaluation = px.evaluations.submit(
    model="model.drawer-dp@1.0.0",
    matrix="evaluation.drawer-heldout@1.0.0",
    backend="isaac_lab",
    seed=100_000,
)
report = px.evaluations.wait(evaluation).report
```

## 指标

至少覆盖：

| 维度 | 指标示例 |
| --- | --- |
| 任务 | 成功率、阶段完成率、恢复率、超时率 |
| 轨迹 | 路径长度、动作平滑度、执行时间、末端误差 |
| 接触 | 峰值力、冲量、滑移、非目标碰撞、安全触发 |
| 泛化 | 新资产、新场景、新语言和参数区间成功率 |
| 效率 | 推理延迟、吞吐、显存、能耗 |
| 校准 | 仿真—真实指标差、置信区间和显著性 |

聚合指标必须保留分层结果，不能用总体成功率掩盖某一物体或条件下的系统性失败。

## 失败归因

失败类别：

- `perception`：目标、姿态、深度或部件识别错误；
- `planning`：任务分解、路径或可达性错误；
- `control`：跟踪、增益、时延或动作饱和；
- `contact`：抓取、摩擦、碰撞或力控制问题；
- `timeout`：未在预算内完成；
- `data_quality`：传感缺失、时间错位或标注错误；
- `unknown`：证据不足，等待人工复核。

归因记录必须引用失败 Episode、模型、场景、任务和部署版本，并附带证据窗口。

## 部署前门禁

真实机器人部署前必须通过：

1. 模型输入输出模式与机器人适配器兼容；
2. 控制频率、坐标 frame、关节顺序和单位一致；
3. 动作限制、工作空间、速度、力和碰撞安全策略可用；
4. 延迟、丢帧和传感降级测试通过；
5. 仿真评测达到任务门槛；
6. 人工急停、看护模式和回滚策略已配置。

## 部署

```python
deployment = px.deployments.deploy(
    model="model.drawer-dp@1.0.0",
    robot="robot.franka-lab-01",
    task="task.open-drawer@1.0.0",
    safety_profile="manipulation.conservative@1.0.0",
    mode="shadow",
)
```

建议按 `replay → shadow → supervised → production` 逐级放开权限。部署记录保存模型摘要、机器人校准、控制器、ROS 2 图、安全配置和操作者。

## 真机 Episode

真实执行使用与仿真相同的 Episode 合同，`source="real"`。这样可以：

- 直接比较同一任务的仿真和真实信号；
- 将真实失败加入评测与训练视图；
- 反演物理参数和传感噪声；
- 定位 Sim-to-Real 差异来自视觉、动力学还是控制。

## 反馈到再生成

```python
analysis = px.feedback.analyze(
    deployment=deployment,
    episodes="query://deployment.failures",
)

plan = px.feedback.plan_regeneration(
    analysis,
    budget={"episodes": 20_000, "gpu_hours": 100},
)

new_dataset = px.feedback.execute(plan)
```

`RegenerationPlan` 应明确：

- 目标失败簇与证据；
- 需要调整的资产、物理、场景、语言或策略变量；
- 采样分布和优先级；
- 成功/失败/恢复比例；
- 预算、停止条件和验收指标；
- 新数据与原数据的合并策略。

## 闭环停止条件

闭环不是无限重训。达到以下任一条件时应暂停：

- 真实成功率和安全指标达到门槛；
- 新增数据的边际收益低于阈值；
- 失败属于硬件、任务定义或资产不可识别问题；
- 数据预算或真机风险超过上限；
- 仿真—真实差异无法由当前参数族解释，需要升级模型。

每轮闭环产生独立数据集、模型和评测版本，保留完整对比链路。
