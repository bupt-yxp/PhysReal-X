---
title: PhysReal-X
description: 连接数字孪生、物理仿真、多模态数据与机器人学习的统一平台
hide:
  - navigation
  - toc
  - footer
---

<div class="prx-home" markdown>

<section class="prx-hero" markdown>

<div class="prx-hero__copy" markdown>

<span class="prx-eyebrow">具身智能数据与实验平台</span>

# 把真实资产、仿真数据与机器人学习连成一条生产链

PhysReal-X 统一管理数字孪生、场景任务、仿真执行、Episode 数据、策略训练、独立评测与真机反馈，让每一次实验都可运行、可追溯、可复现。

<div class="prx-actions" markdown>

[快速开始](getting-started/quickstart.md){ .md-button .md-button--primary }
[浏览 API](api/index.md){ .md-button }

</div>

<div class="prx-hero__proof" role="list">
  <div role="listitem"><strong>统一契约</strong><span>资产、任务、数据与模型共享稳定语义</span></div>
  <div role="listitem"><strong>多后端</strong><span>按能力接入主流仿真器、训练器与机器人</span></div>
  <div role="listitem"><strong>全血缘</strong><span>版本、摘要、种子、环境与结果完整关联</span></div>
</div>

</div>

<div class="prx-hero__visual" role="img" aria-label="PhysReal-X 连接真实世界、数字孪生、仿真数据和机器人学习">
  <div class="prx-hero__halo" aria-hidden="true"></div>
  <img src="assets/images/physreal-x.png" alt="PhysReal-X 项目标识">
  <div class="prx-hero__node prx-hero__node--asset"><span>01</span>数字孪生</div>
  <div class="prx-hero__node prx-hero__node--data"><span>02</span>仿真数据</div>
  <div class="prx-hero__node prx-hero__node--policy"><span>03</span>机器人策略</div>
  <div class="prx-hero__caption"><strong>一套平台契约</strong><span>贯通本地、集群与真实机器人</span></div>
</div>

</section>

<nav class="prx-entry-strip" aria-label="文档快捷入口">
  <a href="getting-started/"><span>01</span><strong>入门</strong><small>安装并完成首个项目</small></a>
  <a href="user-guide/"><span>02</span><strong>用户指南</strong><small>按工作流使用平台</small></a>
  <a href="api/"><span>03</span><strong>API 文档</strong><small>查询服务与领域模型</small></a>
  <a href="development/"><span>04</span><strong>开发</strong><small>实现适配器与扩展能力</small></a>
</nav>

<section class="prx-section">

<div class="prx-section__heading">
  <span>平台能力</span>
  <h2>覆盖从真实世界到真机反馈的完整链路</h2>
  <p>每个环节都使用可验证的资源、明确的生命周期和可追溯的输入输出。</p>
</div>

<div class="prx-capability-grid">
  <a class="prx-capability-card" href="user-guide/assets/">
    <span class="prx-card-index">01</span>
    <h3>资产与数字孪生</h3>
    <p>接入 RGB-D、点云、CAD、Mesh、USD 与 Gaussian，统一描述几何、关节、碰撞、物性和标定。</p>
    <strong>查看资产工作流 <span aria-hidden="true">→</span></strong>
  </a>
  <a class="prx-capability-card" href="user-guide/scenes-and-tasks/">
    <span class="prx-card-index">02</span>
    <h3>场景与任务</h3>
    <p>组合机器人、物体、传感器与环境，声明观测、动作、初始条件、成功条件和随机化空间。</p>
    <strong>查看场景与任务 <span aria-hidden="true">→</span></strong>
  </a>
  <a class="prx-capability-card" href="user-guide/simulation/">
    <span class="prx-card-index">03</span>
    <h3>仿真与运行编排</h3>
    <p>根据能力选择 Isaac Sim、Genesis、MuJoCo 等后端，统一提交、观察、取消、恢复和重试。</p>
    <strong>查看运行方式 <span aria-hidden="true">→</span></strong>
  </a>
  <a class="prx-capability-card" href="user-guide/data-generation/">
    <span class="prx-card-index">04</span>
    <h3>多模态数据生成</h3>
    <p>同步记录图像、深度、点云、接触、力觉、状态与动作，发布经过质量门禁的 Episode。</p>
    <strong>查看数据生成 <span aria-hidden="true">→</span></strong>
  </a>
  <a class="prx-capability-card" href="user-guide/training/">
    <span class="prx-card-index">05</span>
    <h3>训练与独立评测</h3>
    <p>用一致入口组织模仿学习、强化学习、VLA 和世界模型训练，并在冻结矩阵上独立评测。</p>
    <strong>查看训练流程 <span aria-hidden="true">→</span></strong>
  </a>
  <a class="prx-capability-card" href="user-guide/evaluation-and-deployment/">
    <span class="prx-card-index">06</span>
    <h3>部署与反馈闭环</h3>
    <p>记录真机运行与失败证据，定位模型、数据和物理参数缺口，驱动校准与定向数据再生成。</p>
    <strong>查看闭环方法 <span aria-hidden="true">→</span></strong>
  </a>
</div>

</section>

<section class="prx-section prx-section--workflow">

<div class="prx-section__heading">
  <span>标准工作流</span>
  <h2>一次定义，贯穿生成、训练与部署</h2>
  <p>资源在进入运行前解析为精确版本；结果通过质量检查后才发布，失败不会留下可见的半成品。</p>
</div>

<div class="prx-workflow" aria-label="PhysReal-X 标准工作流">
  <div><span>01</span><strong>注册资产</strong><small>结构、物理与标定</small></div>
  <div><span>02</span><strong>定义任务</strong><small>观测、动作与目标</small></div>
  <div><span>03</span><strong>提交运行</strong><small>后端、策略与种子</small></div>
  <div><span>04</span><strong>发布数据</strong><small>Episode 与数据集</small></div>
  <div><span>05</span><strong>训练评测</strong><small>模型、指标与失败</small></div>
  <div><span>06</span><strong>真机回流</strong><small>校准与定向补数</small></div>
</div>

</section>

<section class="prx-sdk" markdown>

<div class="prx-sdk__copy" markdown>

<span class="prx-eyebrow">统一 Python SDK</span>

## 同一套接口，连接本地工作区与远程平台

服务按领域组织，业务代码不直接依赖数据库、任务队列或某个仿真器。将端点从本地切换到远程时，资源模型和方法语义保持一致。

[阅读快速开始](getting-started/quickstart.md) · [查看客户端约定](api/client.md)

</div>

<div class="prx-sdk__code" markdown>

```python
from physrealx import PhysRealX

px = PhysRealX.connect("local:///.physrealx")

run = px.runs.submit(
    task="open-drawer@1.0.0",
    scene="drawer-lab@1.0.0",
    backend="isaac-sim@5",
    episodes=100,
    seed=42,
)

dataset = px.datasets.create_from_run(run.id)
```

</div>

</section>

<section class="prx-final-cta" markdown>

<span class="prx-eyebrow">开始构建</span>

## 用一个完整项目认识 PhysReal-X

从安装环境开始，注册资产、定义场景任务、生成多模态数据，并把结果交给训练与评测服务。

[安装与环境](getting-started/installation.md){ .md-button }
[开始第一个项目](getting-started/first-project.md){ .md-button .md-button--primary }

</section>

</div>
