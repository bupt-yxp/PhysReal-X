# 03 · 场景与任务

读取模块二资产，布置 oven4 与 input/linear_stage.json 推杆，编译接触关闭烤箱门任务。推杆行程按实际接触区域计算；位置控制增益显式设为 800 N/m。机器人控制与物体成功条件分开定义，专家轨迹经过 MuJoCo 实际执行检查。

在现有 PhysReal-X 环境中打开 [demo.ipynb](demo.ipynb)，选择 PhysReal-X 内核并全部运行。先完成上游模块；`python -m examples.run_all` 会自动按顺序执行六个模块。

模板不是通用语言规划；静态布局不是机械臂可达性证明。

Notebook 将业务资源引用导出到 `output/results.json`；执行脚本另存 `execution.json` 与运行副本，失败停止下游。详细对照见 [功能验收](../../docs/acceptance.md)。
