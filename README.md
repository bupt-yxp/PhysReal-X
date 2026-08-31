# PhysReal-X

[中文文档](https://bupt-yxp.github.io/PhysReal-X/) ·
[快速开始](https://bupt-yxp.github.io/PhysReal-X/getting-started/quickstart/) ·
[用户指南](https://bupt-yxp.github.io/PhysReal-X/user-guide/) ·
[API 文档](https://bupt-yxp.github.io/PhysReal-X/api/)

PhysReal-X 是面向具身智能的仿真、数据与机器人学习一体化平台。它位于物理仿真器、
生成式模型和真实机器人之上，通过统一资源模型把以下环节组织成可复现闭环：

```text
已采集真实数据 → 数字孪生 → 场景与任务 → 仿真及多模态数据 → 策略训练与评测 → 真机反馈
```

PhysReal-X 不重复实现物理引擎。SAPIEN、Isaac Sim、Genesis、渲染器、训练框架和机器人
运行时都通过能力明确的适配器接入；平台负责资源编排、方法执行契约、数据契约、版本来源与关联追踪、
数据质量检查和不可变运行证据。

## 文档结构

文档站只保留四个稳定入口：

- **首页**：平台定位、能力边界与业务闭环。
- **入门**：安装、快速开始和第一个可复现契约实验。
- **用户指南**：资产、场景、仿真、数据、训练、评测与部署教程。
- **API 文档**：Python SDK、领域模型、CLI、错误和幂等约定。

## 快速开始

PhysReal-X 核心 SDK 支持 Python 3.8–3.12；完整 ArtGS 方法使用项目统一的 Python 3.10、
PyTorch 2.4.1 与 CUDA 12.1 环境：

```bash
conda env create -f environment.yml
conda activate PhysReal-X
python -m pip install --no-build-isolation -e '.[artgs,dev,docs,demo,train]'
physrealx doctor
```

已有同规格环境可直接按需安装内嵌 ArtGS 方法，不创建第二个 Conda/venv：

```bash
python -m pip install --no-build-isolation -e '.[artgs]'
```

训练扩展在 Python 3.9–3.12 使用 NumPy 1.26.4。NumPy 1.26 不支持 Python 3.8，
因此 Python 3.8 环境使用同为 NumPy 1.x 的 1.24.4。

公共 SDK 使用按领域划分的服务接口：

```python
from physrealx import PhysRealX
from physrealx.reference import reference_asset, reference_run, reference_scene, reference_task

px = PhysRealX.open(".physrealx")
asset = reference_asset()
scene = reference_scene(asset)
task = reference_task(scene)
px.assets.register(asset)
px.scenes.register(scene)
px.tasks.register(task)

run = px.runs.submit(reference_run(scene, task, episode_count=8))
dataset = px.datasets.get(px.runs.wait(run).dataset)
```

这段代码使用确定性 Reference backend，只验证领域契约、不可变资源和运行生命周期；
它不是真实物理仿真、`cabinet2` 数字孪生或真机闭环的实验结果。

完整教程见[第一个可复现契约实验](https://bupt-yxp.github.io/PhysReal-X/getting-started/first-project/)，
公共服务、领域模型、方法协议与错误语义见 [API 文档](https://bupt-yxp.github.io/PhysReal-X/api/)。

## 六模块 Demo

[`examples/`](examples/) 按用户指南的六个业务模块提供中文 Jupyter Demo。
方法名与参数直接写在对应 API 单元格中，无需在 Notebook 与 YAML 之间往返查找：

```console
python -m jupyter lab examples/real_assets_and_digital_twin/demo.ipynb
```

每个 Notebook 逐功能说明公开 SDK、输入、产物和质量门禁；未配置证据或未通过探针的
方法会返回 `BLOCKED`，不会生成占位结果。

## 当前实验证据边界

平台已实现不可变资源、方法配方和生命周期门禁；内嵌 ArtGS 仍须使用米制相机、增强深度和
留出真值才能发布合格数字孪生。`cabinet2` 原始包缺少可核验的米制 Marker、完整物性轨迹和
真机 Episode，因此不能宣称完整链路完成。详见
[真实资产与数字孪生指南](docs/user-guide/real-assets-and-digital-twin.md)与
[数据管理与模型训练指南](docs/user-guide/data-management-and-training.md)。

另有一组[历史实际训练证据](experiments/results/cabinet2-training/README.md)：归仓前的 ArtGS 实验完成
35,000 次资产优化，LeRobot ACT 完成 600 次反向传播，动作条件世界模型完成 71 轮训练，
并保存数据快照、模型权重和 50 个固定闭环评测结果。该实验使用工程先验 MuJoCo 抽屉模型，
只证明当时的训练代码确实执行；它不是当前内嵌 `artgs` 方法的合格输出，也不改变
真实物性与真机阶段的 `BLOCKED` 状态。

## 架构原则

- **依赖倒置**：领域层只依赖协议，仿真器、存储、训练框架和设备位于适配器层。
- **统一资源模型**：资产、场景、任务、数据集、模型、评测和方法操作都具有稳定引用。
- **领域服务入口**：用户只通过 `assets`、`scenes`、`tasks`、`runs`、`datasets` 等任务级接口组合流程。
- **场景显式组合**：任务绑定场景实体，运行固定场景与任务版本，不再把单个资产隐式当作完整环境。
- **可复现优先**：每个结果绑定运行参数、输入版本、制品标识、随机种子和运行环境。
- **原子发布**：产物完成校验后才发布清单；失败作业不会留下可见的半成品资源。
- **显式能力协商**：提交前验证后端是否支持所需传感器、物理特性和执行模式。
- **执行证据统一**：项目内方法使用 `OperationSpec`，结果使用 `OperationManifest`。
- **算法源码归仓**：开源方法以固定来源、许可证和修改清单进入 `algorithms/`，不依赖工作区外部 clone。
- **统一可选环境**：方法依赖由根项目 extras 管理；`.[artgs]` 安装到当前 PhysReal-X Conda 环境。`--no-build-isolation` 仅让 CUDA 扩展复用该环境已安装的 Torch，不会创建第二个运行环境。
- **进程故障隔离**：训练可由同一解释器启动子进程隔离 CUDA 故障，但不建立第二套 Python 环境。
- **内容幂等**：方法、精确输入与期望输出相同时返回原结果，不允许同身份覆盖。
- **无隐式副作用**：导入核心包不会初始化 CUDA、仿真器、网络连接或工作区。
- **按需依赖**：基础安装保持跨 Python 版本可用，GPU 依赖只在用户选择对应 extra 时安装。

## 文档开发

```bash
python -m pip install -e '.[docs]'
python -m mkdocs serve
python -m mkdocs build --strict
```

推送到 `main` 后，`.github/workflows/docs.yml` 会严格构建文档并通过 GitHub Actions 发布
GitHub Pages。

## 质量门禁

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest -m 'not gpu' --cov=physrealx --cov-branch
python -m build
python -m mkdocs build --strict
```

## 许可证

PhysReal-X 自有代码使用 MIT，见 [LICENSE](LICENSE)。内嵌算法保留上游许可；
ArtGS 的 Gaussian rasterizer 与 Simple-KNN 仅允许非商业研究/评估，详见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
