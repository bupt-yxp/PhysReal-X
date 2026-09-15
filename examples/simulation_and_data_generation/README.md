# 04 · 多模态物理仿真

读取模块三任务/轨迹，生成双相机八种输出、刚体位姿与外力/力矩、随机化的成功/失败集，并执行推杆接触被动烤箱门。

在现有 PhysReal-X 环境中打开 [demo.ipynb](demo.ipynb)，选择 PhysReal-X 内核并全部运行。先完成上游模块；`python -m examples.run_all` 会自动按顺序执行六个模块。

推杆来自模块三。20 条专家接触轨迹与 6 条不接近物体的失败轨迹均记录固定 8 秒，保留到达目标后的动力学；数据来源为 simulator。另以实际被动关节位置驱动同一资产中已配准的 ArtGS 外观，单独显示该物体。事件仍为帧间亮度阈值模型。

Notebook 将业务资源引用导出到 `output/results.json`；执行脚本另存 `execution.json` 与运行副本，失败停止下游。详细对照见 [功能验收](../../docs/acceptance.md)。
