# 运行与数据生成 API

运行服务编排仿真、策略和记录器，将一组完整 Episode 原子发布为数据集。

## `runs.submit`

```python
px.runs.submit(
    spec: RunSpec,
    *,
    idempotency_key: str | None = None,
    priority: int = 0,
    validate_only: bool = False,
) -> RunRef
```

提交时执行：资源解析、插件版本固定、能力匹配、预算检查和执行计划编译。

`validate_only=True` 返回已编译计划但不创建运行，适合 CI 和成本预检。

## `runs.get`

```python
px.runs.get(run: RunLike) -> RunRecord
```

`RunRecord.status`：

```text
pending → queued → provisioning → running → finalizing → succeeded
                                      ├──────────────→ failed
                                      └──────────────→ cancelled
```

## `runs.wait`

```python
px.runs.wait(
    run: RunLike,
    *,
    timeout: float | None = None,
    poll_interval: float = 2.0,
) -> RunResult
```

成功结果包含数据集精确引用；失败抛出 `RunFailedError`，异常对象带终态记录和结构化错误。

## `runs.watch`

```python
px.runs.watch(
    run: RunLike,
    *,
    cursor: str | None = None,
) -> Iterator[RunEvent]
```

断线后使用最近 cursor 恢复事件流。

## `runs.cancel`

```python
px.runs.cancel(
    run: RunLike,
    *,
    reason: str,
    mode: Literal["graceful", "immediate"] = "graceful",
) -> RunRecord
```

- `graceful`：完成当前原子 Episode 写入后停止；
- `immediate`：请求工作节点尽快终止，未完成 Episode 丢弃。

## `runs.resume`

```python
px.runs.resume(
    run: RunLike,
    *,
    retry_policy: RetryPolicy | None = None,
) -> RunRef
```

从已发布 Episode 继续，保留原逻辑索引；不覆盖原失败记录。不可恢复的规格或能力错误不能 resume。

## `runs.episodes`

```python
px.runs.episodes(
    run: RunLike,
    *,
    success: bool | None = None,
    failure_category: str | None = None,
    limit: int = 100,
    cursor: str | None = None,
) -> Page[EpisodeSummary]
```

## `runs.load_episode`

```python
px.runs.load_episode(
    run: RunLike,
    episode_id: str,
    *,
    signals: Sequence[str] | None = None,
) -> Episode
```

`signals` 可投影高带宽模态，减少读取量；Episode 元数据和时间语义始终完整。

## 示例：提交并监控

```python
run = px.runs.submit(spec, idempotency_key="drawer-generate-v1")

for event in px.runs.watch(run):
    if event.kind == "progress":
        print(f"{event.completed}/{event.total}")

result = px.runs.wait(run)
print(result.dataset)
```

## RunSpec 关键字段

| 字段 | 说明 |
| --- | --- |
| `scene` / `task` | 精确场景与任务引用 |
| `backend` / `policy` | 插件名称与可选版本约束 |
| `episodes` / `seed` | Episode 数量和根随机种子 |
| `parallelism` | 期望并行度 |
| `record` | 需要发布的观测与状态信号 |
| `dataset` | 输出数据集 ID 与版本 |
| `quality_gates` | Episode 与数据集级质量条件 |
| `budget` | 时间、GPU、存储、失败数和成本上限 |
| `tags` | 实验检索标签，不影响执行语义 |

## 主要错误

- `CapabilityMismatchError`：后端或策略能力不足；
- `RunFailedError`：运行进入失败终态；
- `BudgetExceededError`：达到明确预算上限；
- `EpisodeValidationError`：轨迹违反数据合同；
- `WorkerUnavailableError`：没有满足约束的工作节点。
