# 02 · 物理资产构建

读取模块一的真实重建几何，以 VLM 初始化逐部件物性，再执行视频约束、碰撞检查、仿真检查和 USD 导出。当前 VLM 方法为 Qwen3-VL，`start/end` 范围固定。

在 PhysReal-X Conda 环境中运行 [demo.ipynb](demo.ipynb)。输出由 `output/results.json` 索引。

首次推理需指定 VLM 的 Python、模型和版本；已导出的预测仅在重建几何与图像一致时复用。

可视化包括 RGB/Depth/Mask 视频、视觉/碰撞体动画和物理检查图。若局部重建残片被误分到可动部件，流程会同步修正视觉网格与碰撞代理并重新验收，关节范围保持不变。当前视频已含逐帧 Mask；视频表面修正仍需相机位姿。证据格式见[模块说明](../../docs/user-guide/physical-assets.md)。
