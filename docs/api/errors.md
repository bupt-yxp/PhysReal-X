# 错误、状态与幂等

所有预期错误继承 `PhysRealXError`，包含稳定 `code`、可读 `message` 和结构化 `details`。调用方根据异常类型或 `code` 处理，不解析消息文本。

## 错误层级

```text
PhysRealXError
├── DomainValidationError
├── SourcePackageError
├── AssetBuildError
├── ResourceNotFoundError
├── ResourceConflictError
├── ArtifactIntegrityError
├── InvalidStateTransitionError
└── PluginError
    └── PluginCompatibilityError
```

Pydantic 模型构造失败直接抛出 `pydantic.ValidationError`；多个已合法领域对象组合失败时抛出 `DomainValidationError`。

## 捕获错误

```python
from physrealx.errors import AssetBuildError, SourcePackageError

try:
    source = px.assets.ingest(path, **options)
    result = px.assets.build(spec)
except SourcePackageError as exc:
    print(exc.code, exc.details.get("issue_codes"))
except AssetBuildError as exc:
    print(exc.code, exc.details.get("issue_codes"))
```

## 重试规则

| 错误 | 默认是否重试 |
| --- | --- |
| 校验、冲突、能力不匹配 | 否；修改输入 |
| 源包、构建契约或插件错误 | 否；修复输入或插件 |
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
