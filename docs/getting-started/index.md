---
title: 平台概览
description: 了解 PhysReal-X 的定位、完整工作流、核心服务与资源模型
hide:
  - toc
---

# 平台概览

<div class="prx-page-lead" markdown>

PhysReal-X 是面向具身智能的**物理可信数据与机器人学习平台**。它将真实资产、数字孪生、场景任务、物理仿真、多模态数据、策略训练、评测部署与真机反馈组织成一条统一流水线。

[安装平台](installation.md){ .md-button .md-button--primary }
[查看快速开始](quickstart.md){ .md-button }

</div>

## 平台定位

如果 SAPIEN、Genesis、MuJoCo 和 Isaac Sim 主要解决“虚拟世界如何运行”，PhysReal-X 解决的是：

> **机器人需要什么世界、如何批量生成可训练交互数据、如何训练与评测策略，以及如何让真机失败重新驱动数据生成。**

因此，PhysReal-X 位于物理引擎之上，不重新实现碰撞检测、刚体求解或渲染器。

## 平台完整闭环

```text
真实采集
  ↓
数字孪生与物理资产
  ↓
场景组合与任务编译
  ↓
物理仿真、轨迹生成与多模态渲染
  ↓
Episode 校验、数据集发布与导出
  ↓
模仿学习 / 强化学习 / VLA / 世界模型
  ↓
独立评测与真实机器人部署
  ↓
失败归因、参数校准与定向再生成
  └───────────────────────────↺
```

## 六项核心服务

| 服务 | 输入 | 输出 |
| --- | --- | --- |
| 真实资产接入 | RGB-D、点云、多视角图像、机器人位姿、力觉 | 带标定与血缘的采集数据 |
| 数字孪生与资产库 | 采集数据、Mesh、USD、Gaussian、物性测量 | 可交互、可校准、可训练资产 |
| 场景与任务 | 资产、机器人、语言任务、初始/目标状态 | 场景规格、任务规格、轨迹与随机化计划 |
| 仿真与数据生成 | 场景、任务、后端、策略、种子 | 同步多模态 Episode 和数据集 |
| 训练与评测 | 数据集、训练配置、评测矩阵 | 模型、指标、失败案例和对比报告 |
| 部署与反馈 | 策略、机器人配置、真机日志 | 部署记录、失败归因和再生成计划 |

## 四个稳定面

<div class="prx-layer-stack">
  <div class="prx-layer"><strong>用户入口</strong><span>Python SDK、命令行、远程 API 与 Web 控制台，共享同一应用语义。</span></div>
  <div class="prx-layer"><strong>控制面</strong><span>资源注册、能力匹配、任务编译、运行编排、状态管理、评测与再生成计划。</span></div>
  <div class="prx-layer"><strong>计算面</strong><span>隔离运行重建、仿真、轨迹、训练、世界模型与真实机器人适配器。</span></div>
  <div class="prx-layer"><strong>数据面</strong><span>保存内容寻址制品、Episode 时间线、数据集、权重、指标与完整血缘。</span></div>
</div>

## 核心资源

| 资源 | 回答的问题 |
| --- | --- |
| `AssetSpec` | 世界里有什么？几何、结构、关节和物理属性是什么？ |
| `SceneSpec` | 哪些资产和机器人以什么位姿、材质与环境条件组合？ |
| `TaskSpec` | 要完成什么？如何观测、控制、判定成功和终止？ |
| `RunSpec` | 用哪个后端、策略、版本、种子与并行规模运行？ |
| `Episode` | 每个时间点看到了什么、执行了什么、发生了什么？ |
| `DatasetManifest` | 哪些 Episode 组成一个完整且可训练的数据版本？ |
| `ModelManifest` | 模型由哪个数据集、训练器和配置产生？ |
| `EvaluationReport` | 模型在什么条件下表现如何，失败属于哪类？ |
| `DeploymentRecord` | 哪个模型部署到哪台机器人，使用了什么安全与控制配置？ |
| `RegenerationPlan` | 失败暴露了什么数据缺口，需要补充哪些样本？ |

所有资源使用不可变版本与内容摘要引用。任务、数据集、模型和评测报告不会只记录一个易变路径。

## 面向谁

### 平台使用者

使用统一客户端完成资产注册、任务编排、仿真数据生成、模型训练、评测和部署。

### 能力开发者

实现仿真后端、资产构建器、轨迹策略、训练器、数据导出器或机器人驱动，将外部能力转换为平台标准对象。

### 平台维护者

维护领域契约、运行状态机、制品一致性、调度、权限、可观测性和共享契约测试。

## 继续阅读

<div class="prx-route-grid">
  <a class="prx-route-card" href="installation/"><strong>安装与环境</strong><span>准备 Python、核心服务、计算工作节点与存储。</span></a>
  <a class="prx-route-card" href="quickstart/"><strong>快速开始</strong><span>通过 Python SDK 跑通资产、任务、运行、训练和评测。</span></a>
  <a class="prx-route-card" href="first-project/"><strong>第一个完整项目</strong><span>从项目结构和 YAML 规格开始构建可复现工程。</span></a>
  <a class="prx-route-card" href="../user-guide/"><strong>用户指南</strong><span>按照真实工作顺序深入使用平台的每项服务。</span></a>
</div>
