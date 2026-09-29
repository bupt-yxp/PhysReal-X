# 评测、部署与反馈 API

## 独立评测

`evaluations.submit(resource_id=..., version=..., model=..., run=RunSpec, episode_count=..., base_seed=...)` 使用模型精确推理方法实际执行引擎；返回报告 VersionedRef。`evaluations.get_report(ref)` / `evaluations.list_reports()` 读取成功率、失败案例和数据集引用。

示例在新随机种子上验证同一 PartNet 笔记本任务，不能解读为未见资产或现场机器人泛化。

## 遥测、诊断与回流

```python
from physrealx.integrations.feedback import DeploymentEvidence, FailureQuery

source = px.artifacts.put_json(DeploymentEvidence(dataset=dataset_ref))
archive = px.telemetry.record(source, method="episode_telemetry")
telemetry_ref = archive.output("telemetry")
failures = px.failures.analyze(telemetry_ref, method="trajectory_diagnostics")
query = px.artifacts.put_json(
    FailureQuery(report=failures.output("failure_report"), resource_id="partnet.failures")
)
feedback = px.feedback.build_dataset(query, method="failure_episodes")
```

全部返回 `OperationManifest`。原始 Episode 与任务用于重新计算指标；篡改遥测不能成为证据。反馈保留失败标签、种子与原始制品，输出再生成计划；模块六实际在同初态执行专家轨迹并发布新增成功集。

## 设备边界

`deployments.stage` 是外部机器人适配器的执行入口，目前未注册现场 ROS 2 方法；`robot_agent.py` 提供接口/限位/租约/看门狗协议。缺少设备、机器人资产与原始日志时，调用契约不能成为真实部署完成的证据。

规范化实机 Episode 可以进入遥测/反馈，其来源保留 `real`。模拟记录保持 `simulator`，诊断类别属于记录症状，因果归因与物性标定需要独立干预或测量。详见[反馈指南](../user-guide/evaluation-and-feedback.md)。
