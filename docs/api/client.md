# 客户端与公共约定

## `PhysRealX.connect`

```python
PhysRealX.connect(
    endpoint: str,
    *,
    project: str | None = None,
    token: str | None = None,
    timeout: float = 30.0,
    retries: int = 3,
) -> PhysRealX
```

创建轻量客户端，不初始化 GPU、仿真器或训练框架。

| 参数 | 说明 |
| --- | --- |
| `endpoint` | `local:///PATH` 或 `https://HOST` |
| `project` | 远程平台的逻辑项目；本地端点可省略 |
| `token` | 直接令牌或 `env:VARIABLE_NAME` 引用 |
| `timeout` | 单次控制面请求超时，单位秒 |
| `retries` | 仅对安全或带幂等键的请求重试 |

客户端支持上下文管理：

```python
with PhysRealX.connect("local:///.physrealx") as px:
    print(px.health())
```

## `health`

```python
px.health(*, deep: bool = False) -> HealthReport
```

- `deep=False`：检查客户端、元数据和制品服务；
- `deep=True`：同时探测队列、工作节点和制品读写。

## `close`

```python
px.close() -> None
```

幂等关闭网络连接、线程池和本地句柄，不停止远程运行。

## 资源引用

所有服务接受三种引用：

```python
"drawer.real-001@1.0.0"  # 人可读字符串

VersionedRef(...)  # 结构化引用
ResolvedRef(..., digest="sha256:…")  # 精确引用
```

读操作可以使用可读引用；提交运行、训练、评测或部署时，控制面会解析为精确摘要并保存到不可变计划。

## 分页

```python
page = px.assets.list(limit=100)

for item in page.items:
    print(item.ref)

if page.next_cursor:
    next_page = px.assets.list(limit=100, cursor=page.next_cursor)
```

公共分页对象：

```python
Page[T](items: tuple[T, ...], next_cursor: str | None)
```

游标是不可解释字符串，调用方不能拼接或推断内容。

## 幂等键

```python
run = px.runs.submit(spec, idempotency_key="paper-exp-2026-08-26")
```

相同项目、方法和幂等键：

- 请求内容相同：返回原资源或任务；
- 请求内容不同：抛出 `IdempotencyConflictError`；
- 键过期：按新请求处理，过期策略由服务配置声明。

## 异步任务

构建资产、校准、运行、导出、训练和评测返回 `JobRef` 或其子类型：

```python
job = px.assets.build(...)
result = px.jobs.wait(job, timeout=3600)
```

通用方法：

```python
px.jobs.get(job: JobLike) -> JobRecord
px.jobs.wait(job: JobLike, *, timeout: float | None = None) -> JobResult
px.jobs.watch(job: JobLike, *, cursor: str | None = None) -> Iterator[JobEvent]
px.jobs.cancel(job: JobLike, *, reason: str) -> JobRecord
```

## 日志与事件

`watch()` 返回结构化事件，而不是拼接文本：

```python
for event in px.jobs.watch(job):
    match event.kind:
        case "progress":
            print(event.completed, event.total)
        case "metric":
            print(event.name, event.value)
        case "warning":
            print(event.code, event.message)
```

日志文本可通过 `px.jobs.logs()` 获取；事件用于程序控制。

## 配置优先级

从低到高：

```text
系统默认 < 项目配置 < 环境变量 < 客户端参数 < 方法参数
```

每个任务的最终解析配置保存在运行清单中，避免只记录用户原始片段。

## 线程与进程安全

- 客户端实例可在同一进程的多个线程中共享；
- 不保证 fork 后继续复用连接；子进程应重新 `connect()`；
- 领域模型不可变，可安全缓存和跨线程传递；
- 数据流和文件句柄必须显式关闭或使用上下文管理器。
