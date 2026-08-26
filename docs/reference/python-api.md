# Python 接口

`physrealx` 顶层包暴露稳定的应用门面和核心领域模型。导入顶层包不会初始化 GPU、仿真器或工作区。

## 应用门面

```python
from pathlib import Path

from physrealx import PhysRealX

platform = PhysRealX.open(Path(".physrealx"))
```

主要方法：

| 方法 | 作用 |
| --- | --- |
| `open` | 组合本地仓库、制品库和插件注册表 |
| `register_asset` | 注册不可变资产版本 |
| `register_task` | 解析资产引用并注册任务 |
| `execute` | 执行 `RunSpec` |
| `train` | 使用 `TrainingSpec` 训练并注册模型 |
| `evaluate` | 使用独立种子生成评测报告 |
| `load_asset` | 加载并验证资产内容 |
| `load_task` | 加载并验证任务内容 |
| `load_dataset` | 加载数据集清单 |
| `load_model` | 加载模型清单 |
| `load_evaluation` | 加载评测报告 |
| `episodes` | 读取某次运行的全部 Episode |

## 公开领域类型

```python
from physrealx import (
    ActionSpec,
    ArtifactRef,
    AssetSpec,
    DatasetManifest,
    DataValue,
    Episode,
    EpisodeStep,
    EvaluationReport,
    ModelManifest,
    RunRecord,
    RunSpec,
    SignalSpec,
    TaskSpec,
    TrainingSpec,
    VersionedRef,
)
```

这些类型均使用 Pydantic v2 严格校验。规格模型被冻结并拒绝未知字段，适合在进程边界和制品清单中直接序列化。

## 注册与运行

```python
from physrealx.reference import reference_asset, reference_run, reference_task

asset = reference_asset()
task = reference_task(asset)

asset_ref = platform.register_asset(asset)
task_ref = platform.register_task(task)
record = platform.execute(reference_run(asset, task, episode_count=8))

print(asset_ref.digest)
print(task_ref.digest)
print(record.status)
```

## 加载运行数据

```python
episodes = platform.episodes(record.run_id)

for episode in episodes:
    print(episode.index, episode.success, len(episode.steps))
```

完整训练与评测示例见[数据生成、训练与评测](../user-guide/workflow.md)。第三方能力的接入原则见[插件与适配器](../concepts/plugins.md)。
