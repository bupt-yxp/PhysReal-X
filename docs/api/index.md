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

以下命名空间统一公共语义，不代表所列算法依赖已随核心包安装。方法型操作只会调用当前项目
已登记且探针通过的适配器。

| 命名空间 | 主要操作 |
| --- | --- |
| `px.methods` | `list`、`inspect`、`installations`、`bind`；查询方法身份与真实就绪状态 |
| `px.operations` | 读取和列出能力方法的不可变执行证据 |
| `px.assets` | 观测导入、数字孪生、关节、碰撞、物性、封装与验收 |
| `px.scenes`、`px.tasks`、`px.trajectories` | 场景生成/组合、任务编译/校验和轨迹生成 |
| `px.simulations`、`px.renders`、`px.events`、`px.pointclouds` | 物理轨迹与严格同步的多模态派生 |
| `px.runs` | 提交运行、读取状态、等待结果和加载 Episode |
| `px.episodes`、`px.datasets` | Episode 封装、数据归一化、快照与读取 |
| `px.training`、`px.models`、`px.world_models` | 训练、模型登记和动作条件 rollout |
| `px.evaluations`、`px.deployments`、`px.telemetry` | 冻结评测、分级部署和 MCAP 留证 |
| `px.failures`、`px.feedback` | 证据化失败归因与新的定向数据集 |
| `px.artifacts` | 标识符寻址制品的原子写入、读取与大小检查 |

## 可执行契约参考链

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

上例使用内置确定性 Reference backend 和小型 `torch_bc` 回归基线，仅用于检验公共
API、资源引用和原子发布；外部仿真、实际训练与真机能力以已安装方法及其证据为准。

## 稳定性规则

- 公共模型冻结、拒绝未知字段、NaN/Inf、非法单位和未解析引用；
- 方法必须固定能力、版本、源码 revision、许可证、环境、输入输出契约和可选权重制品；
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
