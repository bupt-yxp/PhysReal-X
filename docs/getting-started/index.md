# 平台概览

PhysReal-X 用统一资源和数据格式连接六个模块。入口是 Python SDK、命令行和可执行 Notebook；网页用于展示 Demo 与查阅文档。

<div class="prx-route-grid">
  <a class="prx-route-link" href="demos/"><strong>先看效果 · 六模块可视化</strong><span>重建、碰撞、机械臂、多模态数据、世界模型和反馈闭环；点击动画可查看原尺寸。</span></a>
  <a class="prx-route-link" href="quickstart/"><strong>开始实验 · Jupyter 入门</strong><span>准备环境、选择内核、打开 Notebook、逐步运行并查看保存的结果。</span></a>
</div>

## 数据流程

| 数据线 | 当前示例 |
| --- | --- |
| 实拍资产 | laptop1 RGB-D → 数字孪生 → VLM 先验、几何修正、碰撞与 USDZ |
| 机器人任务 | PartNet-Mobility → 房间与 Panda → 仿真 Episode → 训练 → 评测反馈 |

[查看六模块输入输出](../user-guide/index.md) · [运行 Demo](quickstart.md)

## 核心资源

| 资源 | 内容 |
| --- | --- |
| `AssetSpec` | 几何、关节、物性与证据 |
| `SceneSpec` / `TaskSpec` | 场景布局、语言任务、观测、动作和成功条件 |
| `RunSpec` / `Episode` | 后端与策略配置、同步观测和执行结果 |
| `DatasetManifest` / `DatasetSnapshot` | 数据版本、划分和来源 |
| `ModelManifest` / `EvaluationReport` | 权重、训练配置、独立评测和失败记录 |

领域服务编排方法适配器，SQLite 保存注册信息，不可变制品库存放网格、图像、轨迹和权重。核心导入不初始化仿真器或 GPU。

## 方法与后端

算法通过 `method` 选择；物理仿真通过 `backend="mujoco"` 或 `backend="isaac_sim"` 选择。依赖检查与执行结果分别记录，未安装的方法明确失败。

[安装与环境](installation.md) · [方法目录](../api/capability-methods.md)
