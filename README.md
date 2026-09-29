# PhysReal-X

从真实物体重建、物理资产与场景任务，到多模态仿真、机器人学习和评测反馈。Python SDK、Notebook 与网页共享项目实现。

[六模块与可视化](https://bupt-yxp.github.io/PhysReal-X/) · [安装](docs/getting-started/installation.md) · [API](docs/api/index.md)

## 运行 Demo

```bash
conda activate PhysReal-X
python -m ipykernel install --sys-prefix --name physreal-x --display-name PhysReal-X
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python -m examples.run_all
```

也可在 Jupyter 中打开下列 Notebook，选择 **PhysReal-X** 内核运行。输入、方法与参数直接在 Notebook 中设置，`output/results.json` 索引实际产物。

| 模块 | 内容与可视化 |
| --- | --- |
| [01 数字孪生](examples/real_assets_and_digital_twin/demo.ipynb) | laptop1 实拍 RGB-D、部件与关节重建；外观/重建/参考模型同屏动画 |
| [02 物理资产](examples/physical_asset_building/demo.ipynb) | 模块一真实几何、VLM 物性先验、视频约束、视觉与碰撞几何修正、仿真检查、USDZ |
| [03 场景任务](examples/scene_and_task_generation/demo.ipynb) | PartNet-Mobility 房间、官方 Panda、语言任务、接触轨迹与 Blender 动画 |
| [04 仿真数据](examples/simulation_and_data_generation/demo.ipynb) | MuJoCo/Isaac Sim 执行、同步 RGB-D/Mask/事件/点云/位姿/力觉与多模态动画 |
| [05 数据训练](examples/data_management_and_training/demo.ipynb) | 快照、ACT/IQL、DINO-WM、官方 VLA/世界模型训练接口与可视化 |
| [06 评测反馈](examples/evaluation_and_feedback/demo.ipynb) | 失败诊断、恢复示范、补数、再训练、独立评测与闭环对比动画 |

实际依赖：**01 → 02**（实拍 laptop1），**03 → 04 → 05 → 06**（PartNet 资产库）。模块三当前独立选取资产。完整环境、输入和模型要求见[安装说明](docs/getting-started/installation.md)。

## 架构

```text
Notebook / CLI / 文档站
    ↓
领域服务：资产 · 场景 · 任务 · 仿真 · 数据 · 训练 · 评测 · 反馈
    ↓
统一资源契约、版本与运行生命周期
    ↓
方法适配器 + SQLite 注册表 + 不可变制品库
```

算法按需加载；仿真器用 `backend` 选择，算法用 `method` 选择。核心导入不启动 GPU、仿真器或网络。内嵌算法保留在 `algorithms/`，不另建一套业务资源。

VLM 输出为物性先验，仿真可行性不替代独立实测；OpenVLA-OFT/TD-MPC2 当前提供训练交接接口。真机日志缺失时不声明完成真机闭环。

## 开发验证

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy
MUJOCO_GL=egl python -m pytest -m 'not real_data' --cov=physrealx --cov-branch --cov-fail-under=80
python -m pip check
python -m build
python -m mkdocs build --strict
```

文档站使用 MkDocs。运行 Demo 后用 `python docs/_hooks/demo_gallery.py --refresh` 更新展示快照及来源清单，再执行严格构建；构建无需本地原始数据或模型。

自有代码为 MIT；内嵌算法遵守各自许可，见[第三方声明](THIRD_PARTY_NOTICES.md)。
