# 六模块 Jupyter 验收

激活现有 PhysReal-X Conda 环境，在项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python -m examples.run_all
```

也可按下表顺序打开 Notebook，选择 **PhysReal-X** 内核，点击全部运行。
所有业务 API 和参数直接可见；`run_all.py` 仅负责 Jupyter 执行和回执保存。

| 顺序 | Notebook | 输入 |
| --- | --- | --- |
| 1 | [真实资产与数字孪生](real_assets_and_digital_twin/demo.ipynb) | 本地 cabinet2 实拍 RGB-D 与逆向建模 USD |
| 2 | [物理资产](physical_asset_building/demo.ipynb) | 模块一的 USD 参考模型 |
| 3 | [场景与任务](scene_and_task_generation/demo.ipynb) | 模块二的物理资产 |
| 4 | [多模态仿真](simulation_and_data_generation/demo.ipynb) | 模块三场景/任务/轨迹与显式推杆资产 |
| 5 | [数据与训练](data_management_and_training/demo.ipynb) | 模块四的成功与失败数据；官方 DINOv2 权重首次下载后复用 |
| 6 | [评测与反馈](evaluation_and_feedback/demo.ipynb) | 模块五 ACT 模型及原始失败集 |

同用项目 `.physrealx` 工作区；上游引用通过已完成回执读取，无需手工粘贴制品 ID。
`output/demo.executed.ipynb` 保留实际运行输出，源 Notebook 保持清洁。
总报告为 `output/notebook-acceptance.json`。已有上游结果时可用 `--from-module 4` 从第四模块继续；报告明确记录本次请求的模块范围。

`COMPLETED` 表示所请求软件阶段执行成功，独立物理/现场机器人证据仍由各自字段标识；
异常会保存 `FAILED` 并停止下游。详细要求与证据边界见[功能验收](../docs/acceptance.md)。
