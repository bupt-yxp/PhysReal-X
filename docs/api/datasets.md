# 数据集 API

## `datasets.get`

```python
px.datasets.get(ref: DatasetLike) -> DatasetManifest
```

读取并验证数据集清单。默认不加载 Episode 大制品。

## `datasets.list`

```python
px.datasets.list(
    *,
    task: str | None = None,
    source: Literal["simulator", "real", "world_model"] | None = None,
    tags: Sequence[str] = (),
    limit: int = 100,
    cursor: str | None = None,
) -> Page[DatasetSummary]
```

## `datasets.query`

```python
px.datasets.query(
    dataset: DatasetLike,
    *,
    where: FilterExpression | Mapping[str, JSONValue],
    signals: Sequence[str] | None = None,
    view_id: str | None = None,
    version: str = "1.0.0",
) -> DatasetView
```

查询生成不可变逻辑视图，不复制底层制品。过滤表达式、信号投影和来源摘要进入视图清单。

## `datasets.split`

```python
px.datasets.split(
    dataset: DatasetLike,
    *,
    by: str | Sequence[str],
    ratios: Mapping[str, float],
    seed: int,
    stratify: Sequence[str] = (),
) -> DatasetSplits
```

所有 ratio 必须为正且和为 1。`by` 指定防泄漏实体键。

## `datasets.merge`

```python
px.datasets.merge(
    datasets: Sequence[DatasetLike],
    *,
    resource_id: str,
    version: str,
    conflict: Literal["reject", "deduplicate"] = "reject",
    weights: Mapping[str, float] | None = None,
) -> ResolvedRef[DatasetManifest]
```

合并要求信号语义兼容；`deduplicate` 仅按 Episode 摘要去重，不进行近似视觉去重。

## `datasets.validate`

```python
px.datasets.validate(
    dataset: DatasetLike,
    *,
    level: Literal["manifest", "sample", "deep"] = "sample",
    checks: Sequence[str] | None = None,
    sample_size: int = 1000,
    seed: int = 0,
) -> DatasetQualityReport
```

## `datasets.open`

```python
px.datasets.open(
    dataset: DatasetLike,
    *,
    split: str | None = None,
    signals: Sequence[str] | None = None,
    streaming: bool = True,
    prefetch: int = 2,
) -> DatasetReader
```

读取器支持：

```python
with px.datasets.open(dataset, split="train") as reader:
    for episode in reader.episodes(shuffle=True, seed=17):
        ...

    for batch in reader.batches(size=64, window=8):
        ...
```

## `datasets.export`

```python
px.datasets.export(
    dataset: DatasetLike,
    *,
    format: str,
    destination: str,
    split: str | None = None,
    config: Mapping[str, JSONValue] | None = None,
) -> JobRef[ExportManifest]
```

导出器名称示例：`lerobot-v3`、`rlds`、`parquet-mp4`、`zarr`、`hdf5`、`rosbag2`。

## `datasets.stats`

```python
px.datasets.stats(
    dataset: DatasetLike,
    *,
    group_by: Sequence[str] = (),
    metrics: Sequence[str] = ("count", "success_rate", "duration"),
) -> DatasetStatistics
```

## `datasets.diff`

```python
px.datasets.diff(left: DatasetLike, right: DatasetLike) -> DatasetDiff
```

比较 Episode 集、信号模式、来源、任务分布、质量和统计，不读取全部高带宽制品。

## 主要错误

- `DatasetSchemaError`：信号或 Episode 模式不一致；
- `DatasetLeakageError`：切分违反防泄漏键；
- `DatasetQualityError`：质量门失败；
- `ExportCompatibilityError`：目标格式无法表达源语义；
- `ArtifactIntegrityError`：数据制品损坏。
