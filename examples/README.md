# 六模块 API 示例

激活现有 PhysReal-X Conda 环境，在项目根目录执行（可编辑安装使各 Notebook 目录都能直接导入 `examples._display`）：

```bash
python -m pip install -e . --no-deps --no-build-isolation
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python -m examples.run_all
```

也可按下表顺序打开 Notebook，选择 **PhysReal-X** 内核，点击全部运行。
Notebook 直接展示业务 API、参数和结果；`_display.py` 统一负责环境初始化、路径定位、绘图和表格。

各 Notebook 通过 `PROJECT_ROOT, OUTPUT_DIR = setup_notebook("模块目录")` 初始化；不再修改 `sys.path`、重复设置环境变量或检查解释器。初始化尊重外部已设置的环境变量，并核对 SDK 是否来自当前源码。方法、训练数量等业务参数直接写在 API 调用处。

Notebook 负责输入、显式参数、API 编排和展示；数据采样、几何计算、信号契约、权重获取和导出一致性由项目实现。
相机/RGB-D 的 `max_frames_per_stream=None` 表示使用全部视图，整数表示采样预算；省略参数保留接口默认值。全量超过项目安全上限会报错，不静默截断。
场景执行器由 `px.scenes.place_linear_actuator` 放置，相机由 `CameraSpec.look_at` 定向，传感器契约由 `px.tasks.with_sensors` 生成，编码器由 `px.world_models.prepare_encoder` 获取和校验。

更新项目源码或依赖后，先点击 **Restart（重启内核）**，再点击 **全部运行**。
重新加载 Notebook 文件或清除输出不会更新内存中已导入的 SDK 类；执行脚本每次使用新内核。

| 顺序 | Notebook | 输入 |
| --- | --- | --- |
| 1 | [真实资产与数字孪生](real_assets_and_digital_twin/demo.ipynb) | 本地 oven4 实拍 RGB-D-Mask 与逆向建模 USD |
| 2 | [物理资产](physical_asset_building/demo.ipynb) | 模块一的 USD 参考模型 |
| 3 | [场景与任务](scene_and_task_generation/demo.ipynb) | 模块二的物理资产 |
| 4 | [多模态仿真](simulation_and_data_generation/demo.ipynb) | 模块三场景/任务/轨迹与显式推杆资产 |
| 5 | [数据与训练](data_management_and_training/demo.ipynb) | 模块四的成功与失败数据；官方 DINOv2 权重首次下载后复用 |
| 6 | [评测与反馈](evaluation_and_feedback/demo.ipynb) | 模块五策略模型、数据快照及评测任务 |

同用项目 `.physrealx` 工作区；每个 Notebook 最后通过 `px.artifacts.export(OUTPUT_DIR, results, files=...)` 发布输出：项目先暂存并校验文件，最后发布 `results.json` 完成标记。索引仅保存业务资源引用和下游所需配置。后续模块直接解析并交给 SDK 校验，无需读取验收报告或手工粘贴 ID。
`run_all.py` 独立负责执行验收：`output/demo.executed.ipynb` 保留实际输出，`output/execution.json` 记录单元格状态、耗时和异常；源 Notebook 保持清洁。
总报告为 `output/notebook-acceptance.json`。已有上游结果时可用 `--from-module 4` 从第四模块继续；续跑前会校验所有上游 Notebook 的源码哈希、代码单元、执行副本和结果索引，报告同时记录本次执行范围和六模块有效证据。任一上游证据过期都会拒绝续跑。

`COMPLETED` 表示本次请求的所有代码单元执行并导出结果；跳过、异常、超时或执行中修改源码均不能通过。执行器开始时移除本次及下游旧结果索引，失败停止下游；同一工作区的并发验收会被拒绝。
手动在 Jupyter 中执行时，结果索引表示最后一次成功导出；修改输入后应按顺序全部运行。物性与真机证据以业务资源的来源字段为准。详细要求与证据边界见[功能验收](../docs/acceptance.md)。

当前 oven4 六模块执行状态见各模块 `output/execution.json`，汇总状态见 `output/notebook-acceptance.json`。
