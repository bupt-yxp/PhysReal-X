---
title: PhysReal-X 文档
description: 从真实资产、物理仿真到机器人学习闭环的统一数据与实验平台
hide:
  - toc
---

<div class="prx-intro" markdown>

<img src="assets/images/physreal-x.png" alt="PhysReal-X 项目标识">

<div markdown>

# 从真实资产，到可复现的机器人学习闭环

PhysReal-X 是位于 Isaac Sim、Genesis、MuJoCo 与真实机器人之上的具身智能数据与实验平台。它用统一契约连接数字孪生、场景任务、仿真执行、Episode 数据、策略训练、评测与失败回流。

[认识平台](getting-started/index.md){ .md-button .md-button--primary }
[运行完整闭环](getting-started/quickstart.md){ .md-button }

</div>

</div>

!!! important "先看清平台边界"
    PhysReal-X **不重新实现物理引擎**，也不把多个论文仓库拼成一个大工程。它定义稳定的数据对象、执行协议和血缘规则，再让仿真器、重建算法、世界模型、训练器与机器人驱动以独立适配器接入。

## 一条完整主链路

<div class="prx-flow">
  <div class="prx-flow__step"><span class="prx-flow__number">01 · REAL</span><strong>真实采集</strong><small>RGB-D、点云、事件、力觉、机器人状态与标定信息</small></div>
  <div class="prx-flow__step"><span class="prx-flow__number">02 · ASSET</span><strong>数字孪生</strong><small>Mesh、USD、Gaussian、部件、关节、约束与物性</small></div>
  <div class="prx-flow__step"><span class="prx-flow__number">03 · SIM</span><strong>场景与仿真</strong><small>任务编排、轨迹、物理执行、随机化与同步渲染</small></div>
  <div class="prx-flow__step"><span class="prx-flow__number">04 · DATA</span><strong>Episode 数据</strong><small>观测—动作—状态时间线、标签、事件与不可变数据集</small></div>
  <div class="prx-flow__step"><span class="prx-flow__number">05 · LEARN</span><strong>训练与评测</strong><small>模仿学习、强化学习、VLA、世界模型与独立评测</small></div>
  <div class="prx-flow__step"><span class="prx-flow__number">06 · REAL</span><strong>真机反馈</strong><small>失败归因、参数校准、缺口定位与定向再生成</small></div>
</div>

这条链路中，资产、任务、运行、Episode、数据集、模型和评测报告都具有明确版本与摘要。由此可以回答：**用了什么、怎么生成、由谁训练、在哪失败、如何复现。**

## 按你的目标阅读

<div class="grid cards prx-doc-cards" markdown>

-   :material-map-search-outline:{ .lg .middle } **我要理解这个平台**

    ---

    理解平台定位、六项服务、完整业务闭环，以及与底层仿真器的关系。

    [阅读平台概览](getting-started/index.md)

-   :material-play-circle-outline:{ .lg .middle } **我要跑通一个完整实验**

    ---

    从资产注册开始，完成任务编排、数据生成、训练和独立评测。

    [阅读用户指南](user-guide/index.md)

-   :material-puzzle-outline:{ .lg .middle } **我要接入仿真器或算法**

    ---

    先看扩展协议、能力声明、生命周期、失败隔离和验收门槛。

    [阅读插件开发](development/plugin-development.md)

-   :material-api:{ .lg .middle } **我要查接口**

    ---

    查看应用门面、领域模型、运行时协议和完整服务 API。

    [进入 API 文档](api/index.md)

</div>

## 平台交付面

| 能力域 | 统一能力 |
| --- | --- |
| 领域契约 | 严格、冻结、带 schema 版本的资产、任务、运行、数据集、模型和评测对象 |
| 资源与制品 | 元数据目录、SHA-256 内容寻址制品、不可变版本和完整血缘 |
| 仿真执行 | Isaac Sim/Isaac Lab、Genesis、MuJoCo 等后端的统一会话协议与能力匹配 |
| 多模态数据 | RGB、Depth、Mask、点云、事件、接触、力觉、状态与动作的同步 Episode |
| 训练 | 模仿学习、强化学习、VLA 与世界模型的一致训练、登记和评测入口 |
| 真机闭环 | ROS 2 采集与部署、失败归因、物理校准、缺口定位和定向再生成 |

## 为什么先做文档与契约

仿真器、数据格式和训练框架变化很快；资产、任务、Episode、血缘和失败语义一旦混乱，任何算法接入都会形成不可维护的耦合。先固定平台语言、对象关系、扩展边界和验收标准，再实现各类生产适配器，是成本最低也最稳健的建设顺序。

[查看架构与实施顺序](development/architecture.md){ .md-button .md-button--primary }
