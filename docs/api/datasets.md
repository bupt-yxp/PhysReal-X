# 数据集 API

`DatasetManifest` 是不可变 Episode 清单。`datasets.get(ref)` / `datasets.list()` 读取已登记数据；`runs.episodes(run)` 从制品存储加载并核对 Episode ID、来源运行、成败标签及完整时间线。

## 合并与发布

`datasets.register(manifest)` 验证实际数组、dtype/shape、信号单位、时间戳与动作 Schema，原子登记。

```python
collection = px.artifacts.put_json(
    {
        "schema": "physrealx.episode-collection/1",
        "datasets": [
            expert_dataset.model_dump(mode="json"),
            failure_dataset.model_dump(mode="json"),
        ],
    }
)
operation = px.datasets.ingest(
    collection,
    method="episode_dataset",
    config={"resource_id": "cabinet2.dataset", "version": "1.0.0"},
)
```

返回 `dataset`（DatasetBundle）和 `quality_report`。源数据集必须同 Schema；同 Episode 不能重复混入。原始成功和失败标签保持不变。

## 快照

```python
dataset_bundle = next(item.artifact for item in operation.outputs if item.name == "dataset")
snapshot_operation = px.datasets.snapshot(
    dataset_bundle, method="episode_split", config={"seed": 2026}
)
```

返回核心 `DatasetSnapshot` 与 `snapshot_manifest`，固定数据、代码源树、环境和 `split.train/validation/test`。真实采集按 capture_id，仿真按 scene/task/version/seed 分组；窗口不跨 Episode。快照载入时重新检查原始身份与分组交叉。

这是当前工作区的快照实现，没有 DVC、LeRobot/RLDS/MCAP 转换占位方法。规范 Episode 已保留面向 IL/RL/VLA/世界模型所需的语言任务引用、观测–动作–下一状态与终止信息；外部数据格式需真实导入适配器。

## 专家训练与反馈补充

`datasets.training_subset(snapshot, resource_id=..., required_tag="expert_demonstration", augmentations=(recovery_dataset,))` 发布成功专家训练集。补充数据必须有相同观测/动作 Schema、通过数据质量检查且无重复 Episode；即使重新生成了 Episode ID，也不能与原 validation/test 的采集或场景–任务–种子分组重叠。
