---
title: 用户指南
description: 按完整业务工作流使用 PhysReal-X 生成数据、训练策略并建立真机反馈闭环
hide:
  - toc
---

# 用户指南

<div class="prx-page-lead" markdown>

本指南按照真实工作顺序组织，不要求预先理解平台内部模块。无论底层选择 Isaac Sim、Genesis、MuJoCo 还是真实机器人，资产、场景、任务、运行、Episode、数据集、模型和评测都使用同一套语义。

[从资产开始](assets.md){ .md-button .md-button--primary }
[查看完整项目](../getting-started/first-project.md){ .md-button }

</div>

## 标准工作流

<div class="prx-flow">
  <div class="prx-flow__step"><span class="prx-flow__number">01</span><strong>准备资产</strong><small>采集、导入、重建、结构解析、物理校准与质量检查</small></div>
  <div class="prx-flow__step"><span class="prx-flow__number">02</span><strong>定义场景任务</strong><small>组合机器人和物体，声明观测、动作、初始与成功条件</small></div>
  <div class="prx-flow__step"><span class="prx-flow__number">03</span><strong>生成数据</strong><small>选择后端与策略，批量执行、同步记录和发布 Episode</small></div>
  <div class="prx-flow__step"><span class="prx-flow__number">04</span><strong>管理数据集</strong><small>校验、查询、切分、统计、导出和冻结版本</small></div>
  <div class="prx-flow__step"><span class="prx-flow__number">05</span><strong>训练评测</strong><small>IL、RL、VLA、世界模型训练与独立评测</small></div>
  <div class="prx-flow__step"><span class="prx-flow__number">06</span><strong>真机闭环</strong><small>部署、记录失败、归因、校准和定向再生成</small></div>
</div>

## 使用教程

<div class="prx-route-grid">
  <a class="prx-route-card" href="assets/"><strong>资产与数字孪生</strong><span>把 Mesh、USD、Gaussian 或真实采集转换为可训练资产。</span></a>
  <a class="prx-route-card" href="scenes-and-tasks/"><strong>场景与任务</strong><span>构建环境、编译任务并定义可判定的目标与约束。</span></a>
  <a class="prx-route-card" href="simulation/"><strong>仿真运行与后端</strong><span>选择后端、配置并行执行并管理完整运行生命周期。</span></a>
  <a class="prx-route-card" href="data-generation/"><strong>多模态数据生成</strong><span>生成包含图像、深度、状态、动作、接触和力觉的 Episode。</span></a>
  <a class="prx-route-card" href="datasets/"><strong>数据集管理</strong><span>版本化、查询、切分、质量分析和导出训练数据。</span></a>
  <a class="prx-route-card" href="training/"><strong>策略训练</strong><span>训练模仿学习、强化学习、VLA 与世界模型并登记模型。</span></a>
  <a class="prx-route-card" href="evaluation-and-deployment/"><strong>评测、部署与反馈闭环</strong><span>建立独立评测、真机部署、失败归因和定向再生成闭环。</span></a>
</div>

## 三条使用原则

### 规格进入 Git，制品进入制品库

YAML/JSON 规格、测试和小型配置进入版本控制。图像、点云、USD、Gaussian、Episode、数据集和模型权重通过 `ArtifactRef` 引用，避免大文件和易变路径进入 Git。

### 先校验，再运行

资产图、单位、坐标系、信号模式、后端能力和任务条件应在创建昂贵仿真会话前完成校验。运行时只处理已经编译和解析的规格。

### 失败也是数据

不要只保存成功轨迹。失败 Episode 必须保留完整观测、动作、状态、事件和失败类别，才能用于失败恢复、对比学习、评测和定向补数。

## 资源引用

人可读引用使用：

```text
resource_id@semantic_version
```

执行时平台将其解析为：

```text
resource_type + resource_id + version + sha256 digest
```

同名同版本不能被不同内容覆盖。修改任何影响行为或数据语义的字段时，发布新版本。
