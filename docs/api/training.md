# 训练 API

训练服务统一模仿学习、强化学习、VLA 和世界模型任务。公共 API 不暴露 Slurm、Kubernetes、Ray 或某个训练框架的作业对象。

## `training.submit`

```python
px.training.submit(
    spec: TrainingSpec,
    *,
    idempotency_key: str | None = None,
    priority: int = 0,
    validate_only: bool = False,
) -> TrainingRef
```

提交前检查：数据/环境模式、训练器能力、设备、精度、资源配额、输出模型版本和恢复检查点。

## `training.get`

```python
px.training.get(training: TrainingLike) -> TrainingRecord
```

状态：

```text
pending → queued → provisioning → running → checkpointing → finalizing → succeeded
                                              ├──────────────────────→ failed
                                              └──────────────────────→ cancelled
```

## `training.wait`

```python
px.training.wait(
    training: TrainingLike,
    *,
    timeout: float | None = None,
) -> TrainingResult
```

成功结果包含 `model: ResolvedRef[ModelManifest]`。

## `training.watch`

```python
px.training.watch(
    training: TrainingLike,
    *,
    cursor: str | None = None,
) -> Iterator[TrainingEvent]
```

事件包括进度、指标、资源、检查点、警告和状态变化。

## `training.cancel`

```python
px.training.cancel(
    training: TrainingLike,
    *,
    reason: str,
    save_checkpoint: bool = True,
) -> TrainingRecord
```

若请求保存检查点，状态先进入 `checkpointing`。检查点保存失败不会伪装成正常取消。

## `training.resume`

```python
px.training.resume(
    training: TrainingLike,
    *,
    checkpoint: ArtifactRef | Literal["last"] = "last",
    overrides: Mapping[str, JSONValue] | None = None,
) -> TrainingRef
```

`"last"` 会在提交时解析为精确检查点摘要。只允许覆盖运行资源、日志频率等声明为可恢复配置；改变数据集、模型结构或动作语义需要新训练规格。

## `training.get_model`

```python
px.training.get_model(model: ModelLike) -> ModelManifest
```

## `training.list_models`

```python
px.training.list_models(
    *,
    trainer: str | None = None,
    task: str | None = None,
    tags: Sequence[str] = (),
    limit: int = 100,
    cursor: str | None = None,
) -> Page[ModelSummary]
```

## `training.compare`

```python
px.training.compare(
    models: Sequence[ModelLike],
    *,
    metrics: Sequence[str] = (),
) -> ModelComparison
```

只比较训练与 validation 记录；正式任务能力比较使用评测服务。

## TrainingSpec

```python
TrainingSpec(
    resource_id: str,
    version: str,
    trainer: str,
    seed: int,
    dataset: DatasetLike | None = None,
    environment: EnvironmentSpec | None = None,
    split: str | None = None,
    config: Mapping[str, JSONValue] = {},
    resources: ResourceRequest = ResourceRequest(),
    validation: ValidationSchedule | None = None,
    checkpoint: CheckpointPolicy = CheckpointPolicy(),
    tags: tuple[str, ...] = (),
)
```

`dataset` 与 `environment` 至少一个存在；具体训练器可以要求恰好一个或同时存在。

## 示例

```python
training = px.training.submit(
    TrainingSpec(
        resource_id="model.drawer-dp",
        version="1.0.0",
        trainer="diffusion_policy",
        dataset="dataset.drawer-hybrid@1.0.0",
        split="train",
        seed=17,
        config={"epochs": 200, "batch_size": 256},
        resources={"gpu": 4, "gpu_memory_gb": 24},
    ),
    idempotency_key="drawer-dp-v1",
)
model = px.training.wait(training).model
```

## 主要错误

- `TrainerCompatibilityError`：训练器不支持输入模式或设备；
- `TrainingFailedError`：任务进入失败终态；
- `CheckpointError`：检查点写入、校验或恢复失败；
- `ResourceUnavailableError`：资源请求无法满足；
- `ModelConflictError`：输出模型版本已由不同内容占用。
