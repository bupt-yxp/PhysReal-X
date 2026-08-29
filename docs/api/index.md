---
title: API 总览
description: PhysReal-X Python API 的领域服务、方法契约与使用入口
hide:
  - toc
---

# API 总览

<div class="prx-page-lead" markdown>

PhysReal-X 以不可变领域模型组织资产、场景、任务、运行、Episode、数据集和模型；算法与后端通过
能力匹配的 `MethodSpec` 接入。

[客户端与公共约定](client.md){ .md-button .md-button--primary }
[查看领域模型](models.md){ .md-button }
[能力与方法](capability-methods.md){ .md-button }

</div>

## 创建平台实例

```python
from physrealx import PhysRealX

px = PhysRealX.open(".physrealx")
```

## 服务命名空间

| 命名空间 | 主要操作 |
| --- | --- |
| `px.methods` | `list`、`get`、`bind`、注册外部方法描述符 |
| `px.assets` | 观测预检/导入、数字孪生构建、资产注册与校验 |
| `px.scenes`、`px.tasks` | 注册、读取和列出场景/任务 |
| `px.runs` | 提交运行、读取状态、等待结果和加载 Episode |
| `px.datasets` | 读取和列出已发布数据集 |
| `px.training` | 提交训练、读取和列出模型 |
| `px.evaluations` | 独立评测、读取和列出报告 |
| `px.artifacts` | 内容寻址制品写入、读取与完整性验证 |

## 可执行闭环

```python
from physrealx import MethodCapability, TrainingSpec
from physrealx.reference import (
    reference_asset,
    reference_run,
    reference_scene,
    reference_task,
)

asset = reference_asset()
scene = reference_scene(asset)
task = reference_task(scene)
px.assets.register(asset)
px.scenes.register(scene)
px.tasks.register(task)

run = px.runs.submit(reference_run(scene, task, episode_count=8))
result = px.runs.wait(run)
dataset = px.datasets.get(result.dataset)

trainer = px.methods.bind(
    MethodCapability.POLICY_TRAIN,
    "torch_bc",
    config={"device": "cpu", "epochs": 80},
)
model = px.training.submit(
    TrainingSpec(
        resource_id="drawer.bc",
        version="1.0.0",
        dataset=result.dataset,
        trainer=trainer,
    )
)
```

## 稳定性规则

- 公共模型冻结、拒绝未知字段、NaN/Inf、非法单位和未解析引用；
- 方法必须固定能力、版本、源码 revision、许可证、环境、输入输出契约和可选权重摘要；
- 资源发布后不可变，所有结果可追溯到精确输入与方法；
- 后端对象不穿透服务边界，核心包不隐式加载 GPU 依赖；
- 本地工作区与远程控制面使用不同客户端生命周期，不以端点字符串切换实现。

## 关键参考

<div class="prx-route-grid">
  <a class="prx-route-link" href="client/"><strong>客户端与公共约定</strong><span>工作区、服务边界、方法选择和一致性保证。</span></a>
  <a class="prx-route-link" href="capability-methods/"><strong>能力与方法</strong><span>开源实现、运行边界、许可证和发布门禁。</span></a>
  <a class="prx-route-link" href="models/"><strong>领域模型</strong><span>资源、运行、Episode、数据集、模型和评测对象。</span></a>
  <a class="prx-route-link" href="cli/"><strong>命令行</strong><span>初始化、诊断、校验、运行、训练和演示。</span></a>
</div>
