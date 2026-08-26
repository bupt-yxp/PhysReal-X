# PhysReal-X

[中文文档](https://bupt-yxp.github.io/PhysReal-X/) ·
[快速开始](https://bupt-yxp.github.io/PhysReal-X/getting-started/quickstart/) ·
[用户指南](https://bupt-yxp.github.io/PhysReal-X/user-guide/) ·
[API 文档](https://bupt-yxp.github.io/PhysReal-X/api/)

PhysReal-X 是面向具身智能的仿真、数据与机器人学习一体化平台。它位于物理仿真器、
生成式模型和真实机器人之上，通过统一资源模型把以下环节组织成可复现闭环：

```text
真实采集 → 数字孪生 → 场景与任务 → 仿真及多模态数据 → 策略训练与评测 → 真机反馈
```

PhysReal-X 不重复实现物理引擎。SAPIEN、Isaac Sim、Genesis、渲染器、训练框架和机器人
运行时都通过能力明确的适配器接入；平台负责资源编排、异步作业、数据契约、版本血缘、
质量门禁和可观测性。

## 文档结构

文档站只保留五个稳定入口：

- **首页**：平台定位、能力边界与端到端闭环。
- **入门**：安装、快速开始和第一个完整项目。
- **用户指南**：资产、场景、仿真、数据、训练、评测与部署教程。
- **API 文档**：Python SDK、领域模型、扩展协议、CLI、错误和幂等约定。
- **开发**：架构边界、数据契约、插件开发与质量门禁。

## 快速开始

Python 3.11 是推荐开发环境：

```bash
conda env create -f environment.yml
conda activate PhysReal-X
python -m pip install -e '.[dev,docs]'
physrealx doctor
```

公共 SDK 使用按领域划分的服务接口：

```python
from physrealx import PhysRealX

px = PhysRealX.connect("local:///.physrealx")

asset = px.assets.register("configs/assets/drawer.yaml")
scene = px.scenes.create("configs/scenes/drawer_lab.yaml")
task = px.tasks.create("configs/tasks/open_drawer.yaml")

run = px.runs.submit(
    task=task.ref,
    scene=scene.ref,
    backend="isaac-sim@5",
    episodes=100,
    seed=42,
    wait=True,
)

dataset = px.datasets.create_from_run(run.id, name="drawer-open-v1")
training = px.training.submit(
    dataset=dataset.ref,
    recipe="diffusion-policy@1",
    config="configs/training/diffusion_policy.yaml",
)
evaluation = px.evaluations.submit(
    policy=training.output_policy,
    suite="drawer-generalization@1",
    wait=True,
)
```

完整教程见[第一个完整项目](https://bupt-yxp.github.io/PhysReal-X/getting-started/first-project/)，
所有方法、参数、返回值与异常见 [API 文档](https://bupt-yxp.github.io/PhysReal-X/api/)。

## 架构原则

- **依赖倒置**：领域层只依赖协议，仿真器、存储、训练框架和设备位于适配器层。
- **统一资源模型**：资产、场景、任务、运行、数据集、模型、评测和部署都具有稳定引用。
- **可复现优先**：每个结果绑定配置快照、输入版本、内容摘要、随机种子和运行环境。
- **原子发布**：产物完成校验后才发布清单；失败作业不会留下可见的半成品资源。
- **显式能力协商**：提交前验证后端是否支持所需传感器、物理特性和执行模式。
- **异步作业统一**：长任务采用一致的提交、观察、取消、重试和错误语义。
- **幂等写入**：创建和提交接口支持幂等键，网络重试不会重复生成资源。
- **无隐式副作用**：导入核心包不会初始化 CUDA、仿真器、网络连接或工作区。

更完整的模块边界见[架构与 SOLID 原则](https://bupt-yxp.github.io/PhysReal-X/development/architecture/)。

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
