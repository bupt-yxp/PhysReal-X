# 错误、状态与幂等

所有预期错误继承 `PhysRealXError`，包含稳定 `code`、可读 `message`、结构化 `details` 和可选 `request_id`。调用方根据异常类型或 `code` 处理，不解析消息文本。

## 错误层级

```text
PhysRealXError
├── ClientError
│   ├── ConfigurationError
│   └── ClientClosedError
├── ValidationError
│   ├── AssetGraphError
│   ├── CoordinateFrameError
│   ├── SignalSchemaError
│   ├── TaskCompilationError
│   └── DatasetQualityError
├── ResourceError
│   ├── ResourceNotFoundError
│   ├── ResourceConflictError
│   └── IdempotencyConflictError
├── IntegrityError
│   └── ArtifactIntegrityError
├── CompatibilityError
│   ├── PluginCompatibilityError
│   ├── CapabilityMismatchError
│   └── BackendCompatibilityError
├── JobError
│   ├── RunFailedError
│   ├── TrainingFailedError
│   ├── EvaluationFailedError
│   ├── JobTimeoutError
│   └── InvalidStateTransitionError
├── DeploymentError
│   ├── DeploymentGateError
│   ├── RobotUnavailableError
│   └── SafetyViolationError
└── TransportError
    ├── AuthenticationError
    ├── AuthorizationError
    ├── RateLimitError
    └── ServiceUnavailableError
```

Pydantic 字段校验错误转换为平台 `ValidationError` 时保留字段路径、输入位置和原始错误类型。

## 捕获错误

```python
from physrealx.errors import CapabilityMismatchError, RunFailedError

try:
    result = px.runs.wait(run)
except CapabilityMismatchError as exc:
    print(exc.details["required"], exc.details["available"])
except RunFailedError as exc:
    print(exc.record.status, exc.record.result.error_code)
```

## 重试规则

| 错误 | 默认是否重试 |
| --- | --- |
| 校验、冲突、能力不匹配 | 否；修改输入 |
| 认证、授权、安全事件 | 否；人工处理 |
| 限流 | 按 `retry_after` 退避 |
| 服务暂不可用 | 仅安全读或带幂等键写入 |
| 工作节点失联 | 由调度器按 Episode/检查点边界恢复 |
| 制品摘要错误 | 否；隔离制品并调查来源 |

## 幂等语义

所有资源注册天然幂等：相同身份和相同内容返回同一精确引用；不同内容冲突。

长任务提交通过 `idempotency_key` 保证网络重试不重复创建任务：

```python
px.training.submit(spec, idempotency_key="drawer-dp-v1")
```

取消和关闭也是幂等操作；对已终态任务再次取消返回当前记录，不创建第二个终止事件。

## 资源状态与任务状态

资源只有 `active`、`deprecated`、`retired` 等目录状态，内容始终不可变。异步任务使用明确状态机；非法跳转抛出 `InvalidStateTransitionError`。

## 部分结果

运行失败时：

- 已原子发布的 Episode 可以查询和诊断；
- 未完成或摘要未验证的 Episode 不可见；
- 输出数据集不会发布为成功版本；
- `RunFailedError.record` 提供完成数、失败阶段和可恢复性。

训练、评测、导出和资产构建遵循相同原则：部分制品只有在终态清单引用后才成为正式结果。

## 超时

客户端 `wait(timeout=...)` 超时只停止等待，不取消平台任务：

```python
try:
    px.runs.wait(run, timeout=60)
except JobTimeoutError:
    current = px.runs.get(run)
```

需要取消时显式调用 `cancel()`。

## 安全错误

`SafetyViolationError` 具有最高优先级：

- 不自动重试动作；
- 部署进入受控停止或急停状态；
- 保存触发前后的证据窗口；
- 要求人工确认才能恢复。
