# 05 · 数据与训练

合并模块四数据、分组快照，实际训练官方 ACT 与 DINO-WM 核心，重载闭环评测，执行单帧五秒未来状态/图像和反事实推演。

在现有 PhysReal-X 环境中打开 [demo.ipynb](demo.ipynb)，选择 PhysReal-X 内核并全部运行。先完成上游模块；`python -m examples.run_all` 会自动按顺序执行六个模块。

官方 DINOv2 权重首次下载后在工作区复用。世界模型以 100 步训练展开覆盖五秒预测，使用同一路径训练与推理；检查点仅按验证集各项相对基线误差选择，在测试集数据上分别比较状态、潜在特征、RGB 与保持初态基线；五秒事实和零动作反事实均有对应的 MuJoCo 真值。初始 ACT 基线的实际失败交给模块六闭环修正。

Notebook 将业务资源引用导出到 `output/results.json`；执行脚本另存 `execution.json` 与运行副本，失败停止下游。详细对照见 [功能验收](../../docs/acceptance.md)。
