# 六模块 Demo

```bash
conda activate PhysReal-X
python -m pip install -e . --no-deps --no-build-isolation
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python -m examples.run_all
```

也可打开 Notebook，选择 **PhysReal-X** 内核后全部运行。更新源码后先重启内核。

| 模块 | 输入 | 输出与可视化 |
| --- | --- | --- |
| [01 数字孪生](real_assets_and_digital_twin/demo.ipynb) | laptop1 实拍 RGB-D-Mask 与参考 USD | 重建、配准、质量评估、对比动画 |
| [02 物理资产](physical_asset_building/demo.ipynb) | 模块一真实重建几何与视频 | VLM 物性先验、几何修复候选、碰撞检查、USDZ、资产动画 |
| [03 场景与任务](scene_and_task_generation/demo.ipynb) | PartNet-Mobility、场景类型、自然语言任务 | 场景、任务、轨迹、成功/失败样本与 Panda 操作动画 |
| [04 多模态仿真](simulation_and_data_generation/demo.ipynb) | 模块三场景、任务与轨迹 | Episode、RGB-D/Mask/点云与同步动画 |
| [05 数据与训练](data_management_and_training/demo.ipynb) | 模块四数据集 | 数据快照、ACT/IQL、DINO-WM、外部训练接口与可视化 |
| [06 评测与反馈](evaluation_and_feedback/demo.ipynb) | 模块五模型与评测任务 | 评测报告、恢复示范、更新模型与反馈动画 |

当前依赖是 **01 → 02** 和 **03 → 04 → 05 → 06**。模块三独立选用 PartNet-Mobility 资产。两条流程共用 `.physrealx`，通过各模块 `output/results.json` 传递资源引用。

Notebook 直接展示参数和 API 调用；`_display.py` 负责初始化与显示，`catalog.py` 统一六模块目录和依赖。

`run_all.py` 使用新内核执行，并输出执行副本、逐单元回执和总报告：

- `模块/output/demo.executed.ipynb`
- `模块/output/execution.json`
- `examples/output/notebook-acceptance.json`

使用 `--from-module 3` 可单独运行场景到反馈流程；`--from-module 4` 会先验证模块三及其结果。回执绑定 Notebook、SDK 源码、执行副本、结果文件和实际上游结果的摘要。过期回执拒绝续跑；重跑仅使依赖它的下游结果失效。旧格式回执须重新执行生成。

`COMPLETED` 表示代码执行并导出成功。物性预测、几何修复候选与真机证据仍以各业务报告为准。
