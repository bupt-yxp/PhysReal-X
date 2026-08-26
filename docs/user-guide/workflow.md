# 数据生成、训练与评测

平台把一次实验拆分为可追溯的运行、数据集、模型和评测资源。每个阶段都引用上游资源的精确版本与摘要。

## 执行数据生成

`RunSpec` 固定以下内容：

- 资产和任务引用；
- 仿真后端、策略及其版本；
- 后端与策略配置；
- Episode 数量和基础随机种子；
- 需要记录的信号；
- 目标数据集标识和版本。

```bash
physrealx validate run run.yaml
physrealx run run.yaml --workspace .physrealx
```

执行器会先解析全部引用和插件身份，再创建一个仿真会话与策略实例。每条 Episode 使用独立种子重置，避免为每条轨迹重复构建昂贵场景。

只有请求数量的 Episode 全部存在且通过结构校验后，平台才发布 `DatasetManifest`。中途失败不会产生看似完整的数据集。

## 训练模型

`TrainingSpec` 指定输入数据集、训练器、模型标识、版本和训练配置：

```bash
physrealx validate training training.yaml
physrealx train training.yaml --workspace .physrealx
```

参考 `torch_bc` 训练器读取内联数值观测和动作，使用 PyTorch 训练小型行为克隆网络，并以 Safetensors 保存权重。训练器返回的模型清单必须保持输入数据集、信号模式和插件版本的一致性。

## 独立评测

```bash
physrealx evaluate drawer.bc 1.0.0 run.yaml \
  --id drawer.bc.eval \
  --version 1.0.0 \
  --episodes 16 \
  --base-seed 100000 \
  --workspace .physrealx
```

评测会使用模型对应的推理策略替换原运行策略，并使用新的基础种子生成评测数据集。最终报告包括成功率、Episode 数量以及按类别保存的失败案例。

## 使用 Python 执行完整流程

```python
from pathlib import Path

from physrealx import PhysRealX, TrainingSpec
from physrealx.reference import reference_asset, reference_run, reference_task

platform = PhysRealX.open(Path(".physrealx"), discover_plugins=False)
asset = reference_asset()
task = reference_task(asset)

platform.register_asset(asset)
platform.register_task(task)

base_run = reference_run(asset, task, episode_count=8, base_seed=7)
record = platform.execute(base_run)
assert record.result is not None and record.result.dataset is not None

model = platform.train(
    TrainingSpec(
        resource_id="drawer.bc",
        version="1.0.0",
        dataset=record.result.dataset,
        trainer="torch_bc",
        config={"device": "cuda", "epochs": 120, "seed": 17},
    )
)

evaluation = platform.evaluate(
    resource_id="drawer.bc.eval",
    version="1.0.0",
    model=model,
    run=base_run,
    episode_count=16,
    base_seed=100_000,
)
report = platform.load_evaluation(evaluation)
print(report.success_rate)
```

有关轨迹时间语义和内容寻址规则，请继续阅读[数据契约](../concepts/data-contracts.md)。
