# 评测与真机反馈闭环

本模块先在冻结任务矩阵上评测模型，再逐级部署到真实机器人，保存可回放遥测、确认失败原因，并将确认样本发布为新的训练数据集版本。

## 完成目标

```text
固定基准 → 离线回放 → shadow → 限速/限力 canary → 真机运行
→ MCAP 留证 → 故障归因 → 定向数据集 → 再训练 → 全量回归
```

核心仓库提供评测、部署安全和反馈证据契约，不附带可直接控制真实机器人的 ROS 2 方法。
真实部署需要节点本地 Agent、硬件驱动、机器人规格和现场安全授权。

## 1. 冻结独立评测

```python
report_ref = px.evaluations.submit(
    resource_id="drawer.policy.heldout",
    version="1.0.0",
    model=model_ref,
    run=heldout_run,
    episode_count=200,
    base_seed=100_000,
)

report = px.evaluations.get_report(report_ref)
```

评测套件固定 Episode 顺序、输入记录标识、种子、超时、成功谓词和安全阈值。超时计为失败，
每个 Episode 都必须产生证据；禁止筛选种子或只报告成功轨迹。

策略报告至少包含逐任务成功率及 Wilson 95% 置信区间、完成时间、碰撞/超力/越界、推理 p50/p95/p99 延迟和未见资产最差分组。世界模型还需报告多步状态误差、漂移、接触 F1、规划成功率和反事实动作敏感性；视觉指标只能作为辅助。

实际 cabinet2 工程先验基线的普通/OOD 固定闭环成功率分别为 40% 和 48%。这是真实执行结果，但不等于真实 cabinet2 或真机评测。

## 2. 准备机器人与安全策略

部署输入至少包括：

- URDF/SRDF、关节映射、控制器类型、单位和频率；
- ros2_control 硬件接口与 MoveIt 2 配置；
- 急停、watchdog、工作空间、关节、速度、力矩和末端力限制；
- 模型观测/动作 Schema、归一化和推理延迟预算；
- 统一时钟、TF 树、遥测主题和任务成功谓词。

学习策略只输出期望动作或轨迹，不能绕过确定性安全监督器。

## 3. 分级部署

```python
deployment = px.deployments.stage(
    model_ref,
    robot_interface_ref,
    method=deployment_method,
    config={"mode": "shadow"},
)
```

该调用只生成 `DeploymentPlan/1` 和安全报告，不直接控制硬件。节点本地 Agent 使用租约、心跳和撤销执行计划并发布 `DeploymentRun/1`。状态只能按 `shadow → canary → active` 晋级；急停、心跳超时、陈旧/重放观测、NaN、越界动作或控制周期超时均应立即 fail-closed。

## 4. 记录真机遥测

```python
telemetry = px.telemetry.record(
    deployment_run_ref,
    method=telemetry_method,
)
```

`deployment_run_ref` 必须来自真实节点本地 Agent。MCAP 保存观测、动作、TF、控制器状态、诊断、接触/力、任务谓词和安全事件，并绑定 topic Schema、统一时钟、CRC、模型、机器人及部署计划。

## 5. 分析并确认失败

```python
telemetry_bundle = next(
    output.artifact for output in telemetry.outputs if output.name == "telemetry"
)
failure_report = px.failures.analyze(
    telemetry_bundle,
    method=failure_method,
)
```

确定性规则优先分析传感/同步/标定、感知/姿态/语义、策略无进展/振荡、控制跟踪/超时、接触滑移/卡死/超力/碰撞、任务谓词与基础设施。每条记录包含原因码、发生区间、置信度和证据；无法确认时保留 `unknown`。

视频或世界模型 critic 只能提供二级建议，结论需要规则或人工确认。

## 6. 构建反馈数据集

```python
feedback_dataset = px.feedback.build_dataset(
    confirmed_failure_query,
    method=feedback_method,
)
```

`confirmed_failure_query` 是由规则或人工确认的 `FailureQuery/1`，明确选择故障记录和证据区间。失败、OOD 和人工纠正数据生成新的不可变数据集版本，并保留父 Episode、模型、资产、任务及故障来源；禁止原地修改旧训练集。

再训练只有在独立 failure holdout 改善、旧任务没有退化且安全回归通过时才能晋级。

## 发布门禁

1. 固定基准完整执行，全部失败均有证据；
2. 模型与机器人 Schema、frame、频率和关节顺序一致；
3. 急停、watchdog 与全部硬安全限制测试通过；
4. 推理 p99 延迟低于控制周期，陈旧观测能触发停车；
5. shadow 与 canary 没有关键安全事件；
6. 遥测可完整回放，故障结论具有证据区间；
7. 反馈训练通过独立 holdout 和原有全量回归。

cabinet2 当前没有机器人规格、硬件接口、急停配置和真实 ROS2 遥测，因此真机阶段保持 `BLOCKED`。仿真策略权重或 Mock Agent 不能替代真实部署证据。

完成本模块后，训练数据回到[数据管理与模型训练](data-management-and-training.md)，物性偏差回到[物理可信资产构建](physical-assets.md)。
