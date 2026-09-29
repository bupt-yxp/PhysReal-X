# Jupyter 入门与运行

Jupyter Notebook 把说明、代码和执行结果放在一个 `.ipynb` 文件里。PhysReal-X 的六个 Demo 都可以逐单元运行，直接在代码下方查看图像、动画和指标。

想先了解效果，可打开[六模块可视化](demos.md)。网页展示已保存的结果；实际执行需要本地或服务器上的 Python 环境、输入数据、模型与 GPU。

## 1. 准备环境与数据

在终端进入 **PhysReal-X 项目根目录**（包含 `pyproject.toml` 和 `examples/`），激活已经安装的项目环境：

```bash
conda activate PhysReal-X
python -m pip install -e '.[demo]'
python -m ipykernel install --sys-prefix --name physreal-x --display-name PhysReal-X
```

`demo` 安装 JupyterLab、内核和 Notebook 执行工具，完整重建、仿真、训练依赖见[安装与环境](installation.md)。已有项目环境时直接复用；新机器先按安装文档创建环境。

开始运行前确认：

- 实拍资产数据位于 `examples/real_assets_and_digital_twin/input/ArtModeling/laptop1`。
- 模块三从 `examples/scene_and_task_generation/input/` 下的 PartNet-Mobility 资产库选物体。
- Notebook 配置的模型权重与 Blender 路径可用，CUDA 可访问；数据集和模型不因下载源码就自动齐全。

## 2. 启动 JupyterLab

在项目根目录执行：

```bash
MUJOCO_GL=egl python -m jupyter lab --notebook-dir=examples --ip=127.0.0.1 --port=8888 --no-browser
```

将终端显示的 **带 token 的完整地址**复制到浏览器。左侧文件浏览器中打开 `real_assets_and_digital_twin/demo.ipynb`。`MUJOCO_GL=egl` 用于 Linux GPU 服务器的无窗口渲染，必须在启动内核前设置。

如果 Python 在 SSH 服务器上运行，在 VS Code/Cursor 的 **端口（Ports）** 面板转发服务器 `8888` 端口，然后打开面板提供的本地地址，并使用终端给出的 token。也可在本地终端建立转发：

```bash
ssh -N -L 8888:127.0.0.1:8888 用户名@服务器地址
```

转发建立后，本地浏览器访问 `http://127.0.0.1:8888`。若该端口被占用，以终端实际显示的 Jupyter 端口为准。

## 3. 选择内核并运行第一个单元格

Notebook 右上角选择 **PhysReal-X** 内核。内核决定实际运行代码的 Python；仅看名称不能确认环境，可新增一个代码单元格检查：

```python
import sys
import torch
import physrealx

print("Python:", sys.executable)
print("PhysReal-X:", physrealx.__version__)
print("CUDA:", torch.cuda.is_available())
```

完整 GPU Demo 的 CUDA 检查应为 `True`。若导入失败或解释器路径不对，回到环境安装步骤；不要在错误内核里继续运行。

| 操作 | 用途 |
| --- | --- |
| `Shift + Enter` | 执行当前单元格并移动到下一个 |
| 单元格旁的 `[*]` / 内核忙碌指示 | 正在运行；训练和渲染阶段需要等待 |
| 工具栏停止按钮 / Interrupt Kernel | 中断当前计算，检查错误或修改参数 |
| Restart Kernel and Run All Cells | 从干净内核按顺序运行；更新源码后使用 |
| `Ctrl + S` | 保存 Notebook 及当前可视化输出 |

先阅读配置单元格，确认输入目录、任务描述与后端，再从上到下执行。模块三配置场景与任务；模块四设置 `physics_backend = "mujoco"` 或 `"isaac_sim"`，Isaac Sim 的独立安装见[仿真指南](../user-guide/simulation-and-data-generation.md)。

## 4. 六模块的运行顺序

| 顺序 | `examples/` 下的 Notebook | 前置结果 | 在单元格中查看 |
| --- | --- | --- | --- |
| 01 | `real_assets_and_digital_twin/demo.ipynb` | 实拍 RGB-D 与起止状态 | 外观、重建、参考模型同步动画 |
| 02 | `physical_asset_building/demo.ipynb` | 模块一 | 视频证据、视觉/碰撞几何与验收 |
| 03 | `scene_and_task_generation/demo.ipynb` | PartNet-Mobility 资产库 | 家居、桌面、货架和工业布局与任务动画 |
| 04 | `simulation_and_data_generation/demo.ipynb` | 模块三 | RGB、深度、Mask、事件、点云和物理曲线 |
| 05 | `data_management_and_training/demo.ipynb` | 模块四 | 数据样本、训练评测与世界模型预测 |
| 06 | `evaluation_and_feedback/demo.ipynb` | 模块五 | 失败策略、专家恢复与更新策略同步动画 |

当前依赖是 **01 → 02** 和 **03 → 04 → 05 → 06** 两条数据线。每个模块成功导出 `output/results.json` 后再运行依赖它的模块；失败时先处理报错，避免下游读取旧结果。

## 5. 查看动画与保存的输出

执行展示单元格后，GIF 在代码下方播放，PNG 与指标表直接显示。每个模块的 `output/` 保存动画、面板和 `results.json`，结果清单指向具体制品。

已有批量执行结果时，可直接打开 `output/demo.executed.ipynb` 查看已保存输出，无需重跑训练。手动运行原始 `demo.ipynb` 时使用 `Ctrl + S` 保存自己的输出；它不会自动生成批量执行器的逐单元验收回执。

网页中的[六模块动画](demos.md)来自已完成的执行快照，可点击放大。网页不会随着你在 Jupyter 修改参数而自动更新。

## 6. 一次运行全部模块

在项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python -m examples.run_all --timeout 7200
```

`CUDA_VISIBLE_DEVICES` 按实际设备设置；`--timeout` 为单个代码单元格的超时秒数。该命令实际执行重建、仿真与训练，耗时取决于数据与设备。

执行器按依赖运行并保存每个模块的 `output/demo.executed.ipynb`、`execution.json`、`results.json`，总验收记录在 `examples/output/notebook-acceptance.json`。数据、GPU、模型或 Blender 缺失会明确报错。

六个模块全部执行完成后，可将这轮动画同步到网页：

```bash
python docs/_hooks/demo_gallery.py --refresh
python -m mkdocs serve --dev-addr 127.0.0.1:8000
```

刷新命令会校验执行回执和 Notebook 中实际展示的媒体；未完成的执行不会发布为新快照。终端显示的 `http://127.0.0.1:8000` 是文档网页，`8888` 是 JupyterLab，两者用途不同。

## 常见问题

| 现象 | 处理方法 |
| --- | --- |
| 找不到 PhysReal-X 内核 | 在正确 Conda 环境执行内核注册命令，然后刷新 JupyterLab |
| 修改源码后仍报旧接口错误 | Restart Kernel，然后从第一格重新运行；清除输出不会重启内核 |
| 没有动画或指标 | 检查前置单元格是否完成、是否有异常，再执行展示单元格；也可打开 `output/` 中的 GIF |
| 提示缺少上游结果 | 先按依赖执行上游 Notebook，确认 `results.json` 由本次成功运行生成 |
| Cursor/VS Code 出现 `Error loading renderer` | 这是编辑器渲染模块加载失败；先保存文件并重载窗口，也可在浏览器 JupyterLab 打开同一文件核查 |
| 全部执行完成，但策略门禁为 `REJECT` | 计算流程已完成，候选策略评测不合格；查看模块六面板，补充示范后再评测 |

## 无 GPU 时先熟悉 SDK

可先运行轻量参考契约示例：

```bash
physrealx demo --workspace .physrealx-demo --episodes 8
```

它检查 SDK、资源生命周期与存储，不执行六模块 GPU 重建或仿真。详细步骤见[契约参考实验](first-project.md)。
