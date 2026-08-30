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

因此，PhysReal-X 位于物理引擎之上，不重新实现碰撞检测、刚体求解或渲染器。下述为统一业务链路；每个阶段的可执行性由已注册方法、运行环境和预检证据共同决定。

## 平台业务闭环

```text
已采集真实数据包
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
| 真实资产接入 | 已采集 RGB-D、点云、多视角图像、机器人位姿、力觉 | 带摘要、模态语义和来源记录的观测集 |
| 数字孪生与资产库 | 采集数据、Mesh、USD、Gaussian、物性测量 | 可交互、可校准、可训练资产 |
| 场景与任务 | 资产、机器人、语言任务、初始/目标状态 | 场景规格、任务规格、轨迹与随机化计划 |
| 仿真与数据生成 | 场景、任务、后端、策略、种子 | 同步多模态 Episode 和数据集 |
| 训练与评测 | 数据集、训练配置、评测矩阵 | 模型、指标、失败案例和对比报告 |
| 部署与反馈 | 策略、机器人配置、真机日志 | 部署记录、失败归因和再生成计划 |

## 四个稳定面

<div class="prx-layer-stack">
  <div class="prx-layer-row"><strong>用户入口</strong><span>Python SDK 与命令行共享同一应用语义；远程控制面需独立实现认证和调度边界。</span></div>
  <div class="prx-layer-row"><strong>控制面</strong><span>资源注册、能力匹配、任务编译、运行编排、状态管理、评测与再生成计划。</span></div>
  <div class="prx-layer-row"><strong>计算面</strong><span>隔离运行重建、仿真、轨迹、训练、世界模型与真实机器人适配器。</span></div>
  <div class="prx-layer-row"><strong>数据面</strong><span>保存内容寻址制品、Episode 时间线、数据集、权重、指标与完整来源关联。</span></div>
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

## 适用对象

PhysReal-X 面向具身智能研究人员、机器人算法工程师和数据工程师。用户通过 Python SDK 或命令行管理资源与任务；仿真、训练、评测和真机执行必须选择已安装且通过预检的方法。

## 继续阅读

<div class="prx-route-grid">
  <a class="prx-route-link" href="installation/"><strong>安装与环境</strong><span>准备 Python、核心服务、计算工作节点与存储。</span></a>
  <a class="prx-route-link" href="quickstart/"><strong>快速开始</strong><span>通过 Python SDK 跑通资产、任务、运行、训练和评测。</span></a>
  <a class="prx-route-link" href="first-project/"><strong>第一个可复现契约实验</strong><span>从项目结构和 YAML 规格开始验证平台契约。</span></a>
  <a class="prx-route-link" href="../user-guide/"><strong>用户指南</strong><span>按照真实工作顺序深入使用平台的每项服务。</span></a>
</div>
