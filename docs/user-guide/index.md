---
title: 用户指南
description: 按六个业务模块完成从真实资产到真机反馈的可追溯实验
hide:
  - toc
---

# 用户指南

<div class="prx-page-lead" markdown>

本指南按用户实际完成任务的顺序组织。每个模块说明输入要求、操作步骤、方法选择、输出产物和
质量门禁；底层算法以项目可选依赖安装，并通过已注册且探针通过的方法适配器执行。

[从真实资产开始](real-assets-and-digital-twin.md){ .md-button .md-button--primary }
[查看 API 总览](../api/index.md){ .md-button }

</div>

## 六个业务模块

<div class="prx-flow">
  <div class="prx-flow__step"><span class="prx-flow__number">01</span><strong>真实资产接入与数字孪生</strong><small>导入、标定、深度、几何与关节</small></div>
  <div class="prx-flow__step"><span class="prx-flow__number">02</span><strong>物理可信资产构建</strong><small>碰撞、物性、组装、USD 与验证</small></div>
  <div class="prx-flow__step"><span class="prx-flow__number">03</span><strong>场景与任务生成</strong><small>布局、谓词、随机化与轨迹</small></div>
  <div class="prx-flow__step"><span class="prx-flow__number">04</span><strong>物理仿真与多模态数据生成</strong><small>执行、渲染、传感器与 Episode</small></div>
  <div class="prx-flow__step"><span class="prx-flow__number">05</span><strong>数据管理与模型训练</strong><small>质检、版本、策略、VLA 与世界模型</small></div>
  <div class="prx-flow__step"><span class="prx-flow__number">06</span><strong>评测与真机反馈闭环</strong><small>基准、部署、遥测、归因与回流</small></div>
</div>

## 如何使用本指南

按顺序阅读时，上一个模块的已验证产物就是下一个模块的输入。只处理某一阶段时，可直接进入对应章节，但必须自行提供满足契约的上游资源。

| 模块 | 完成后得到 | 继续阅读 |
| --- | --- | --- |
| 真实资产接入与数字孪生 | 观测集、米制视觉几何和关节模型 | [开始接入](real-assets-and-digital-twin.md) |
| 物理可信资产构建 | 通过验证的物理 USD 资产 | [构建资产](physical-assets.md) |
| 场景与任务生成 | `SceneSpec` 与 `TaskSpec` | [定义任务](scenes-and-tasks.md) |
| 物理仿真与多模态数据生成 | `VerifiedEpisodeCollection` | [生成数据](simulation-and-data-generation.md) |
| 数据管理与模型训练 | 数据集快照和可加载模型包 | [管理与训练](data-management-and-training.md) |
| 评测与真机反馈闭环 | 评测、部署、故障与反馈数据证据 | [评测与回流](evaluation-and-feedback.md) |

## 可执行 Demo

仓库的 `examples/` 为六个模块各提供一个中文 `demo.ipynb`。Notebook 按功能
说明输入、API、产物和门禁；方法与参数直接出现在对应 API 单元格中。输入或方法
不满足条件时返回 `BLOCKED`，不会绕过质量门禁。

```console
python -m jupyter lab examples/real_assets_and_digital_twin/demo.ipynb
```

## 统一操作规则

1. **先检查方法。** `px.methods.inspect(capability, name).ready` 为真，才表示当前工作区具备可执行环境、固定源码或权重及自检证据。
2. **先验证输入。** 路径、方法名或历史结果不能代替版本化资源；单位、坐标系、时间戳和制品引用必须明确。
3. **只发布完整结果。** 阶段通过输出契约与质量门后才发布；失败保留审计记录，但不生成可供下游使用的成功资源。
4. **不覆盖历史版本。** 数据、模型和评测变化时创建新版本，并保留父版本及完整来源。

## 公共底座

资源引用由类型、资源 ID、语义版本和平台生成的制品标识共同确定；图像、点云、USD、Episode 和权重进入制品库，Git 只保存规格、代码和小型清单。

方法选择采用 `capability + method`。核心平台负责契约、编排、证据和发布；算法源码统一置于
`algorithms/`，使用项目可选依赖安装并注册。未安装的方法会明确阻断，不会回退为占位实现。
