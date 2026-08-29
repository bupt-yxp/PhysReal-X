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

- **首页**：平台定位、能力边界与端到端闭环。
- **入门**：安装、快速开始和第一个完整项目。
- **用户指南**：资产、场景、仿真、数据、训练、评测与部署教程。
- **API 文档**：Python SDK、领域模型、CLI、错误和幂等约定。

## 快速开始

PhysReal-X 支持 Python 3.8–3.12；Python 3.11 是推荐运行环境：

```bash
conda env create -f environment.yml
conda activate PhysReal-X
python -m pip install -e '.[dev,docs]'
physrealx doctor
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

完整教程见[第一个完整项目](https://bupt-yxp.github.io/PhysReal-X/getting-started/first-project/)，
所有方法、参数、返回值与异常见 [API 文档](https://bupt-yxp.github.io/PhysReal-X/api/)。

## 架构原则

- **依赖倒置**：领域层只依赖协议，仿真器、存储、训练框架和设备位于适配器层。
- **统一资源模型**：资产、场景、任务、数据集、模型、评测和方法操作都具有稳定引用。
- **领域服务入口**：用户只通过 `assets`、`scenes`、`tasks`、`runs`、`datasets` 等任务级接口组合流程。
- **场景显式组合**：任务绑定场景实体，运行固定场景与任务摘要，不再把单个资产隐式当作完整环境。
- **可复现优先**：每个结果绑定配置快照、输入版本、内容摘要、随机种子和运行环境。
- **原子发布**：产物完成校验后才发布清单；失败作业不会留下可见的半成品资源。
- **显式能力协商**：提交前验证后端是否支持所需传感器、物理特性和执行模式。
- **执行证据统一**：外部 worker 使用 `OperationSpec`，结果使用 `OperationManifest`。
- **算法环境隔离**：重型方法以独立进程或 digest 固定的 OCI worker 安装，核心不导入其依赖。
- **内容幂等**：方法、精确输入与期望输出相同时返回原结果，不允许同身份覆盖。
- **无隐式副作用**：导入核心包不会初始化 CUDA、仿真器、网络连接或工作区。
- **轻量核心**：GPU 算法、仿真器与机器人驱动通过独立适配器接入，不进入基础安装依赖。

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

MIT，见 [LICENSE](LICENSE)。
