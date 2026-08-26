# API 总览

PhysReal-X 提供一个统一客户端和按领域拆分的服务命名空间。相同 API 可连接本地工作区或远程控制面，业务代码不直接依赖存储、队列、仿真器或机器人运行时。

## 连接

```python
from physrealx import PhysRealX

px = PhysRealX.connect("local:///.physrealx")
```

远程平台：

```python
px = PhysRealX.connect(
    "https://physreal.example.org",
    project="embodied-manipulation",
    token="env:PHYSREALX_TOKEN",
)
```

## 服务命名空间

| 命名空间 | 职责 |
| --- | --- |
| `px.assets` | 导入、构建、注册、校准和验证数字资产 |
| `px.scenes` | 组合、注册、验证和物化场景 |
| `px.tasks` | 编译、注册和验证任务 |
| `px.runs` | 提交、观察、取消、恢复仿真和数据生成运行 |
| `px.datasets` | 查询、切分、合并、验证、流式读取和导出数据集 |
| `px.training` | 提交、恢复和观察训练任务，读取模型 |
| `px.evaluations` | 执行冻结矩阵评测和模型对比 |
| `px.deployments` | 部署、停止和审计真实机器人策略 |
| `px.feedback` | 接收反馈、失败归因和创建定向再生成计划 |
| `px.jobs` | 读取所有异步任务的统一状态与事件 |
| `px.workers` | 查看仿真、训练和机器人工作节点及其能力 |
| `px.artifacts` | 上传、下载和验证内容寻址制品 |

## 最小闭环

```python
asset = px.assets.register(AssetSpec.from_file("drawer.asset.yaml"))
scene = px.scenes.register(SceneSpec.from_file("drawer.scene.yaml"))
task = px.tasks.register(TaskSpec.from_file("open_drawer.task.yaml"))

run = px.runs.submit(RunSpec.from_file("generate.run.yaml"))
dataset = px.datasets.get(px.runs.wait(run).dataset)

training = px.training.submit(
    TrainingSpec(dataset=dataset.ref, trainer="diffusion_policy", seed=17)
)
model = px.training.wait(training).model

evaluation = px.evaluations.submit(
    EvaluationSpec(model=model, matrix="drawer.heldout@1.0.0", seed=100_000)
)
report = px.evaluations.wait(evaluation).report
```

## API 稳定性规则

- 顶层只暴露稳定领域模型、客户端和公共错误；
- 服务方法使用关键字参数，避免位置参数含义漂移；
- 资源发布后不可变；更新产生新语义版本；
- 所有长任务返回稳定句柄，可等待、观察、取消和恢复；
- 所有写操作支持幂等键；网络重试不会重复创建资源；
- 所有列表接口使用游标分页，不暴露数据库偏移语义；
- 时间使用带时区 UTC，物理量必须带单位和 frame；
- 不提供后端专属对象作为公共返回值。

## 同步与异步

Python 客户端提供同步 API；异步程序使用 `physrealx.aio` 中的同名客户端：

```python
from physrealx.aio import PhysRealX

async with PhysRealX.connect("https://physreal.example.org") as px:
    run = await px.runs.submit(spec)
    result = await px.runs.wait(run)
```

同步与异步版本共享模型、错误与方法语义，不维护两套领域逻辑。

## 下一步

- [客户端与公共约定](client.md)
- [领域模型](models.md)
- [错误、状态与幂等](errors.md)
