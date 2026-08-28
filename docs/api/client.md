# 客户端与公共约定

Python 客户端是 PhysReal-X 控制面的用户入口。它负责资源操作、任务提交、状态查询和结果读取，不在用户进程中加载仿真器、训练框架或机器人驱动。

```text
Python SDK ──> 平台控制面 ──> 调度器 ──> 仿真 / 训练 / 机器人工作节点
```

本机 GPU 也通过工作节点接入平台，而不是作为另一种客户端端点。这样可以统一鉴权、调度、审计和任务状态，同时保持各后端运行环境隔离。

## 创建客户端

### `PhysRealX.from_profile`

```python
PhysRealX.from_profile(
    name: str = "default",
    *,
    timeout: float | None = None,
    retries: int | None = None,
) -> PhysRealX
```

```python
from physrealx import PhysRealX

px = PhysRealX.from_profile("default")
```

配置档案由命令行创建，业务代码不保存平台地址或访问令牌：

```bash
physrealx profile add default \
  --server https://<平台地址> \
  --project embodied-manipulation
physrealx auth login --profile default
physrealx profile use default
```

| 配置项 | 用途 |
| --- | --- |
| `server` | 平台控制面地址 |
| `project` | 默认资源与任务作用域 |
| `credential` | 系统密钥环或环境变量中的凭据引用 |
| `tls` | 证书验证和企业 CA 配置 |
| `timeout`、`retries` | 请求超时与安全重试策略 |

配置档案可以进入用户配置目录，但凭据本身不得写入项目 YAML、源码或 Git。配置优先级从低到高为：系统默认、配置档案、环境变量、显式客户端参数。

## 生命周期

客户端支持上下文管理；退出时会释放连接池和后台线程，不会停止已经提交的任务：

```python
with PhysRealX.from_profile("default") as px:
    report = px.health()
    print(report.status)
```

### `health`

```python
px.health(*, deep: bool = False) -> HealthReport
```

- `deep=False`：验证身份、项目访问权和控制面连通性；
- `deep=True`：同时检查调度、制品和元数据服务；
- 工作节点能力通过 `px.workers.list()` 查询，不混入客户端连接状态。

### `close`

```python
px.close() -> None
```

`close()` 可重复调用。关闭客户端后再次请求会抛出 `ClientClosedError`。

## 服务边界

客户端按领域暴露服务，不提供底层数据库、队列或仿真器对象：

| 服务 | 主要操作 |
| --- | --- |
| `assets` | 预检/导入已采集数据包，构建、注册和验证数字孪生 |
| `scenes`、`tasks` | 注册、校验、解析版本和编译任务 |
| `runs`、`training`、`evaluations` | 提交、观察、等待、取消和恢复任务 |
| `datasets`、`artifacts` | 查询、验证、流式读取和导出结果 |
| `deployments`、`feedback` | 管理真机运行、反馈与定向补充数据计划 |
| `jobs` | 统一查询全部长任务的状态、事件和日志 |
| `workers` | 查询已注册工作节点的版本、能力和可用状态 |

用户 SDK 与工作节点协议相互独立：SDK 表达“做什么”，工作节点适配器负责“如何在某个后端执行”。增加仿真器或训练框架不会改变用户任务模型。

## 资源引用

读操作可使用人可读引用；提交运行、训练、评测或部署时，平台会解析为带内容摘要的精确引用，并写入不可变任务清单。

```python
"drawer.real-001@1.0.0"  # 人可读引用

VersionedRef(resource_id="drawer.real-001", version="1.0.0")
ResolvedRef(resource_id="drawer.real-001", version="1.0.0", digest="sha256:…")
```

- 语义版本描述资源意图；
- 内容摘要确定实际输入；
- 已发布版本不可覆盖，修改必须生成新版本；
- 返回对象不包含任何后端专属句柄。

## 长任务

构建资产、生成数据、训练、评测和导出均为异步任务。提交方法立即返回稳定句柄，调用方可以退出后重新查询。

```python
run = px.runs.submit(spec, idempotency_key="drawer-generate-v1")

for event in px.jobs.watch(run):
    print(event.kind, event.message)

result = px.jobs.wait(run, timeout=3600)
```

公共任务方法：

```python
px.jobs.get(job: JobLike) -> JobRecord
px.jobs.wait(job: JobLike, *, timeout: float | None = None) -> JobResult
px.jobs.watch(job: JobLike, *, cursor: str | None = None) -> Iterator[JobEvent]
px.jobs.logs(job: JobLike, *, cursor: str | None = None) -> Iterator[LogRecord]
px.jobs.cancel(job: JobLike, *, reason: str) -> JobRecord
```

事件用于程序控制，日志用于诊断；调用方不得解析日志文本判断任务状态。

## 幂等、超时与重试

```python
run = px.runs.submit(spec, idempotency_key="paper-exp-2026-08-28")
```

- 相同项目、方法、幂等键和请求内容返回原任务；
- 相同幂等键对应不同内容时抛出 `IdempotencyConflictError`；
- 客户端只自动重试读请求和带幂等键的安全写请求；
- 单次请求超时不代表平台任务失败，应使用任务句柄继续查询；
- `wait()` 超时只结束等待，不会取消任务；取消必须显式调用 `cancel()`。

## 分页

```python
page = px.assets.list(limit=100)

for asset in page.items:
    print(asset.ref)

if page.next_cursor:
    page = px.assets.list(limit=100, cursor=page.next_cursor)
```

```python
Page[T](items: tuple[T, ...], next_cursor: str | None)
```

游标是不透明字符串，只能原样回传。列表结果具有稳定排序；客户端不得拼接游标或推断数据库偏移。

## 同步与异步

同步客户端适合脚本、Notebook 和常规服务；异步客户端用于已有事件循环的应用：

```python
from physrealx.aio import PhysRealX

async with PhysRealX.from_profile("default") as px:
    run = await px.runs.submit(spec, idempotency_key="drawer-generate-v1")
    result = await px.jobs.wait(run)
```

两种客户端共享领域模型、错误类型和方法语义，不维护两套业务协议。

## 并发与进程

- 同步客户端可在同一进程的多个线程中共享；
- 客户端不能跨 `fork` 复用，子进程必须重新从配置档案创建；
- 领域模型不可变，可安全缓存和跨线程传递；
- 数据流、下载器和上传器必须显式关闭或使用上下文管理器；
- 失败响应统一映射为公共错误类型，详见[错误、状态与幂等](errors.md)。
