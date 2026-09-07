# 评测、部署与反馈 API

## 独立评测

`evaluations.submit(resource_id=..., version=..., model=..., run=RunSpec, episode_count=..., base_seed=...)` 使用模型精确推理方法实际执行引擎；返回报告 VersionedRef。`evaluations.get_report(ref)` / `evaluations.list_reports()` 读取成功率、失败案例和数据集引用。

示例在新随机种子上验证同一 cabinet2 任务，不能解读为未见资产或现场机器人泛化。

## 遥测、诊断与回流

```python
source = px.artifacts.put_json(
    {
        "schema": "physrealx.run-evidence/1",
        "dataset": dataset_ref.model_dump(mode="json"),
    }
)
telemetry = px.telemetry.record(source, method="episode_telemetry")
telemetry_ref = next(item.artifact for item in telemetry.outputs if item.name == "telemetry")
failures = px.failures.analyze(telemetry_ref, method="trajectory_diagnostics")
report_ref = next(item.artifact for item in failures.outputs if item.name == "failure_report")
feedback = px.feedback.build_dataset(
    report_ref,
    method="failure_episodes",
    config={"resource_id": "cabinet2.failures", "version": "1.0.0"},
)
```

全部返回 `OperationManifest`。原始 Episode 与任务用于重新计算指标；篡改遥测不能成为证据。反馈保留失败标签、种子与原始制品，输出再生成计划；模块六实际在同初态执行专家轨迹并发布新增成功集。

## 设备边界

`deployments.stage` 是外部机器人适配器的执行入口，目前未注册现场 ROS 2 方法；`robot_agent.py` 提供接口/限位/租约/看门狗协议。缺少设备、机器人资产与原始日志时，调用契约不能成为真实部署完成的证据。

规范化实机 Episode 可以进入遥测/反馈，其来源保留 `real`。模拟记录保持 `simulator`，诊断类别属于记录症状，因果归因与物性标定需要独立干预或测量。详见[反馈指南](../user-guide/evaluation-and-feedback.md)。
