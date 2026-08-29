# 数据集 API

数据集是不可变 Episode 清单，不是目录别名。当前数据集由成功运行原子发布，用户通过服务读取
清单，并通过运行服务加载经过摘要验证的 Episode。

## `datasets.get`

```python
px.datasets.get(ref: VersionedRef) -> DatasetManifest
```

清单包含 Episode 摘要、来源运行、观测/动作 Schema、总步数、创建时间和方法来源元数据；默认
不加载高带宽 Episode 制品。

## `datasets.list`

```python
px.datasets.list() -> list[VersionedRef]
```

## 加载 Episode

```python
episodes = px.runs.episodes(run)
```

每个摘要必须包含与 Episode JSON 相同的 SHA-256 和制品引用。读取时平台重算领域摘要，并重新
验证时间线、信号模式、动作和终止语义。

## 外部格式与快照

```python
normalized = px.datasets.ingest(raw_mcap, method="lerobot_v3")
snapshot = px.datasets.snapshot(
    normalized.outputs[0].artifact,
    method="dvc",
)
```

`datasets.ingest` 固定输出 `dataset` 和 `quality_report`；`datasets.snapshot` 固定输出
`snapshot` 和 `snapshot_manifest`。两者都发布 `OperationManifest`，不原地修改 `DatasetManifest`。

| 能力 | 推荐方法 |
| --- | --- |
| `dataset.ingest` | `lerobot_v3`、`rlds`、`rosbag2_mcap` |
| `dataset.validate` | `episode_contract` |
| `dataset.snapshot` | `dvc` |

适配器必须发布新的不可变数据集或派生制品，并记录源摘要、转换配置、代码/镜像版本和输出摘要。
训练切分按资产、场景与采集批次防泄漏，不能依赖文件顺序。

## 主要错误

- `ResourceNotFoundError`：数据集或其来源运行不存在；
- `ArtifactIntegrityError`：Episode 制品缺失或摘要不一致；
- `DomainValidationError`：Episode、总步数、来源运行或信号 Schema 不一致。
