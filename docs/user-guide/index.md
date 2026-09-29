---
title: 六模块指南
description: 按 Demo 查看输入、执行流程、输出和可视化。
---

# 六模块指南

各模块的 `demo.ipynb` 声明输入与参数，`output/results.json` 索引实际产物。

| 模块 | 输入 | 输出 |
| --- | --- | --- |
| [01 真实资产与数字孪生](real-assets-and-digital-twin.md) | laptop1 双状态 RGB-D、Mask、参考 USD | 米制相机、部件几何、关节、配准及对比动画 |
| [02 物理资产构建](physical-assets.md) | 模块一重建、实拍视频、VLM | 物性先验、修正几何、碰撞体、USDZ、可行性报告 |
| [03 场景与任务生成](scenes-and-tasks.md) | 场景类型、PartNet-Mobility、自然语言任务 | 场景库、任务脚本库、轨迹库、成功/失败样本索引 |
| [04 物理仿真与多模态数据](simulation-and-data-generation.md) | 模块三场景、任务、机械臂轨迹 | RGB-D、Mask、事件、点云、位姿、动作、接触和 Episode |
| [05 数据管理与模型训练](data-management-and-training.md) | 模块四成功/失败 Episode | 数据快照、ACT/IQL 模型、世界模型、训练接口与评测 |
| [06 评测与反馈](evaluation-and-feedback.md) | 模块五策略、仿真或真机日志 | 失败报告、恢复示范、更新策略、对比动画与更新决策 |

依赖关系为 **01 → 02** 与 **03 → 04 → 05 → 06**。模块三使用独立资产库；当前示例未将模块二的实拍资产接入机器人场景。

[运行方式](../getting-started/quickstart.md) · [API](../api/index.md)
