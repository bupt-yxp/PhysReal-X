# 评测、部署与反馈 API

评测使用冻结模型、任务和独立种子生成报告；真机部署与失败反馈通过能力方法接入，并共享
Episode、制品和方法证据，不共享仿真后端对象。
核心包提供这些领域服务与证据契约，但不附带 LIBERO、ROS 2、MCAP、故障归因或反馈构建
worker；下文方法名均是独立部署候选，只有当前工作区安装且探针通过后才能执行。

## `evaluations.submit`

```python
px.evaluations.submit(
    *,
    resource_id: str,
    version: str,
    model: VersionedRef,
    run: RunSpec,
    episode_count: int = 16,
    base_seed: int = 100_000,
) -> VersionedRef
```

平台检查模型信号 Schema 与任务观测/动作一致，再把运行策略替换为模型清单登记的精确推理
`MethodSpec`。评测身份包含模型、运行规格、输出身份、Episode 数和种子，独立报告不会碰撞。

```python
report_ref = px.evaluations.submit(
    resource_id="drawer.bc.heldout",
    version="1.0.0",
    model=model_ref,
    run=heldout_run,
    episode_count=200,
    base_seed=100_000,
)
```

## `evaluations.get_report` / `evaluations.list_reports`

```python
px.evaluations.get_report(ref: VersionedRef) -> EvaluationReport
px.evaluations.list_reports() -> list[VersionedRef]
```

报告保存模型和评测数据集精确引用、成功率、Episode 数、失败数与结构化失败案例。

## 评测方法

```python
benchmark = px.evaluations.run(model_ref, suite_ref, method="libero")
```

推荐 `libero`、`robotwin2` 和 `simpler` 扩展；每个基准必须固定官方任务、初态、种子、超时和成功
判据，并报告逐任务成功率、Wilson 95% 置信区间、至少三个训练种子和旧版本回归差异。

## 真机部署边界

```python
deployment = px.deployments.stage(model_ref, robot_ref, method="ros2_control")
telemetry = px.telemetry.record(robot_run_artifact, method="rosbag2_mcap")
```

`ros2_control` 方法使用 ros2_control/MoveIt 2 接入机器人，`rosbag2_mcap` 记录观测、动作、TF、诊断、
接触和任务判据。学习策略只输出期望动作或轨迹；急停、工作空间、速度、力矩、力和 watchdog
由确定性安全监督器执行。

## 失败与反馈边界

```python
failure = px.failures.analyze(telemetry.outputs[0].artifact, method="rules_v1")
targeted = px.feedback.build_dataset(
    failure.outputs[0].artifact,
    method="hard_negative",
)
```

候选 `rules_v1` 适配器根据任务时间线、命令—状态误差、接触/力、TF/时钟和诊断产生证据；RoboFAC
只能作为二级 critic。输出至少包含原因代码、起始时刻、置信度、证据引用和确认状态，证据不足时
保留 `unknown`。

`hard_negative` 从确认失败/OOD 切片发布新的 `DatasetRevision`，不得原地修改训练集；提升必须在未
参与反馈构建的 failure holdout 和旧回归集上同时验证。

上述五个接口都返回可追溯 `OperationManifest`。`deployment.stage` 的 `safety_report` 未通过时，
扩展 worker 必须拒绝进入 canary；`failure.analyze` 证据不足时必须输出 `unknown`，不得强制归因。

## 发布门禁

- 评测：逐任务成功率、置信区间、完成时间、碰撞/超力/越界和推理延迟；
- shadow/canary：输入输出 Schema、频率、frame、关节顺序、延迟和安全限制全部通过；
- 真机：关键安全事件为零，并报告干预率、误停率、跟踪误差和故障检测延迟；
- 反馈：故障证据可定位、人工确认可审计、新数据集不可变、旧能力无回归。
