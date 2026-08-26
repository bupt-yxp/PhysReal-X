# 数据集管理

数据集是不可变 Episode 集合与模式清单，不是某个目录的别名。它必须完整描述数据从哪里来、包含什么、如何切分，以及能否被特定训练器消费。

## 数据集清单

`DatasetManifest` 包含：

- 资源 ID、版本、创建时间和摘要；
- Episode 摘要、步数、成功状态和制品引用；
- 来源运行、资产、场景、任务、后端和策略；
- 观测、状态和动作模式；
- 任务、语言和失败标签；
- 统计、质量报告和数据卡；
- 固定 split 与过滤规则；
- 许可证、隐私和使用限制。

## 查询数据

```python
view = px.datasets.query(
    dataset="dataset.open-drawer@1.0.0",
    where={
        "task": "task.open-drawer",
        "success": True,
        "drawer.friction": {"between": [0.25, 0.40]},
    },
    signals=("camera.front.rgb", "robot.state", "action"),
)
```

查询返回不可变数据视图。视图保存来源数据集摘要、过滤表达式和信号投影，可被训练配置引用。

## 固定切分

切分必须基于稳定实体键，而不是依赖文件顺序：

```python
splits = px.datasets.split(
    dataset,
    by="asset_instance",
    ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
    seed=42,
)
```

常见切分维度：物体实例、场景、任务、机器人、语言模板和采集批次。用于泛化评测时，确保同一物体或场景不会通过不同 Episode 泄漏到训练集。

## 质量分析

```python
report = px.datasets.validate(
    dataset,
    level="deep",
    checks=(
        "artifact_integrity",
        "schema_consistency",
        "timestamp_monotonicity",
        "modality_completeness",
        "distribution_coverage",
        "duplicate_detection",
    ),
)
```

质量报告应提供可定位问题的 Episode ID，而不只给一个总分。

## 统计与数据卡

数据卡至少说明：

- 数据生成目的、任务范围和不适用场景；
- 真实、仿真和世界模型数据占比；
- 资产、场景、机器人、语言和失败分布；
- 关键物理参数和随机化范围；
- 模态、采样频率、同步误差与缺失率；
- 质量检查、已知偏差、许可证与隐私限制。

## 合并数据集

```python
merged = px.datasets.merge(
    datasets=(sim_dataset, real_dataset),
    dataset_id="dataset.drawer-hybrid",
    version="1.0.0",
    conflict="reject",
)
```

合并前要求信号语义兼容。若 shape、unit、frame 或动作定义不同，先创建显式转换视图，不进行隐式强制转换。

## 导出训练格式

```python
export = px.datasets.export(
    dataset,
    format="lerobot-v3",
    destination="s3://robot-data/drawer-v1",
    video_codec="h264",
    shard_size="2GB",
)
px.jobs.wait(export)
```

平台可提供 LeRobot、RLDS、Parquet/MP4、Zarr、HDF5 或 ROS bag 适配器。导出物是派生制品，必须保留：来源数据集摘要、导出器版本、转换配置和输出摘要。

## 流式读取

```python
with px.datasets.open(dataset, split="train", streaming=True) as stream:
    for batch in stream.batches(size=64, shuffle=True, seed=17):
        train_step(batch)
```

训练服务使用相同流式读取接口，避免将完整数据集加载到应用进程内存。

## 删除与保留

不可变资源通常不做原地删除。清理流程为：

1. 标记数据集退役；
2. 检查模型、评测、论文结果和部署是否仍引用；
3. 根据保留策略删除无引用派生制品；
4. 保留必要元数据、摘要和审计记录。

不要直接删除对象存储目录；这会绕过引用检查并破坏历史血缘。
