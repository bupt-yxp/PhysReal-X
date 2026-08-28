---
title: API 总览
description: PhysReal-X Python SDK 的服务命名空间、公共约定与完整接口入口
hide:
  - toc
---

# API 总览

<div class="prx-page-lead" markdown>

PhysReal-X 提供统一的平台客户端和按领域拆分的服务命名空间。业务代码面向资产、任务、运行、数据集和模型，不直接依赖存储、队列、仿真器或机器人运行时。

[客户端与公共约定](client.md){ .md-button .md-button--primary }
[查看领域模型](models.md){ .md-button }
[技术能力与方法 API](capability-methods.md){ .md-button }

</div>

## 创建客户端

```python
from physrealx import PhysRealX

px = PhysRealX.from_profile("default")
```

配置档案保存平台地址、当前项目和凭据来源，使连接配置与业务代码分离。计算资源不由客户端直接连接；仿真、训练与机器人运行时以工作节点身份接入平台。

## 服务命名空间

| 命名空间 | 职责 |
| --- | --- |
| `px.assets` | 预检和导入已采集观测，构建、注册并验证数字资产 |
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

async with PhysRealX.from_profile("default") as px:
    run = await px.runs.submit(spec)
    result = await px.runs.wait(run)
```

同步与异步版本共享模型、错误与方法语义，不维护两套领域逻辑。

## 关键参考

<div class="prx-route-grid">
  <a class="prx-route-link" href="client/"><strong>客户端与公共约定</strong><span>连接、资源引用、分页、幂等、异步任务和线程安全。</span></a>
  <a class="prx-route-link" href="models/"><strong>领域模型</strong><span>资产、场景、任务、Episode、数据集、模型和评测对象。</span></a>
  <a class="prx-route-link" href="cli/"><strong>命令行接口</strong><span>项目初始化、资源操作、运行管理、诊断和自动化命令。</span></a>
  <a class="prx-route-link" href="errors/"><strong>错误、状态与幂等</strong><span>错误层次、任务状态机、重试语义和一致性保证。</span></a>
</div>
