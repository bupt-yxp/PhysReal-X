# 评测、部署与反馈 API

## Evaluation API

### `evaluations.submit`

```python
px.evaluations.submit(
    spec: EvaluationSpec | None = None,
    *,
    model: ModelLike | None = None,
    matrix: EvaluationMatrixLike | None = None,
    backend: str | None = None,
    seed: int | None = None,
    idempotency_key: str | None = None,
) -> EvaluationRef
```

可以传完整 `EvaluationSpec`，或使用关键字快速构建。两种形式不能混用冲突字段。

### `evaluations.get` / `wait` / `watch` / `cancel`

```python
px.evaluations.get(ref: EvaluationLike) -> EvaluationRecord
px.evaluations.wait(ref: EvaluationLike, *, timeout: float | None = None) -> EvaluationResult
px.evaluations.watch(ref: EvaluationLike, *, cursor: str | None = None) -> Iterator[EvaluationEvent]
px.evaluations.cancel(ref: EvaluationLike, *, reason: str) -> EvaluationRecord
```

### `evaluations.get_report`

```python
px.evaluations.get_report(report: EvaluationReportLike) -> EvaluationReport
```

### `evaluations.compare`

```python
px.evaluations.compare(
    reports: Sequence[EvaluationReportLike],
    *,
    baseline: EvaluationReportLike | None = None,
    confidence: float = 0.95,
) -> EvaluationComparison
```

比较按同一矩阵轴对齐，输出绝对值、差值、置信区间与失败分布。

## Deployment API

### `deployments.deploy`

```python
px.deployments.deploy(
    *,
    model: ModelLike,
    robot: RobotLike,
    task: TaskLike,
    safety_profile: SafetyProfileLike,
    mode: Literal["replay", "shadow", "supervised", "production"],
    idempotency_key: str | None = None,
) -> DeploymentRef
```

部署前执行输入输出模式、控制频率、坐标、关节、延迟和安全门禁检查。

### `deployments.get`

```python
px.deployments.get(ref: DeploymentLike) -> DeploymentRecord
```

### `deployments.watch`

```python
px.deployments.watch(
    ref: DeploymentLike,
    *,
    cursor: str | None = None,
) -> Iterator[DeploymentEvent]
```

### `deployments.stop`

```python
px.deployments.stop(
    ref: DeploymentLike,
    *,
    reason: str,
    emergency: bool = False,
) -> DeploymentRecord
```

`emergency=True` 请求机器人适配器执行安全急停；是否成功会作为明确事件记录，不能假设网络请求即代表硬件已经停止。

### `deployments.episodes`

```python
px.deployments.episodes(
    ref: DeploymentLike,
    *,
    success: bool | None = None,
    limit: int = 100,
    cursor: str | None = None,
) -> Page[EpisodeSummary]
```

## Feedback API

### `feedback.ingest`

```python
px.feedback.ingest(
    *,
    source: DeploymentLike | DatasetLike | Sequence[EpisodeLike],
    labels: Sequence[FailureLabel] = (),
    notes: str | None = None,
) -> FeedbackBatchRef
```

### `feedback.analyze`

```python
px.feedback.analyze(
    feedback: FeedbackBatchLike | None = None,
    *,
    deployment: DeploymentLike | None = None,
    episodes: EpisodeQuery | None = None,
    analyzer: str = "default",
) -> JobRef[FailureAnalysis]
```

### `feedback.plan_regeneration`

```python
px.feedback.plan_regeneration(
    analysis: FailureAnalysisLike,
    *,
    budget: Mapping[str, JSONValue],
    objective: Sequence[str] = ("success_rate",),
    planner: str = "coverage_gap",
) -> RegenerationPlan
```

### `feedback.execute`

```python
px.feedback.execute(
    plan: RegenerationPlan | RegenerationPlanLike,
    *,
    idempotency_key: str | None = None,
) -> RunRef
```

执行再生成计划返回普通运行句柄，因此共享监控、取消、恢复与数据发布语义。

## 完整示例

```python
evaluation = px.evaluations.submit(
    model=model,
    matrix="evaluation.drawer-heldout@1.0.0",
    backend="isaac_lab",
    seed=100_000,
)
report = px.evaluations.wait(evaluation).report

deployment = px.deployments.deploy(
    model=model,
    robot="robot.franka-lab-01",
    task="task.open-drawer@1.0.0",
    safety_profile="safety.manipulation-conservative@1.0.0",
    mode="supervised",
)

analysis_job = px.feedback.analyze(
    deployment=deployment,
    episodes={"success": False},
)
analysis = px.jobs.wait(analysis_job).result
plan = px.feedback.plan_regeneration(
    analysis,
    budget={"episodes": 20_000, "gpu_hours": 100},
)
new_run = px.feedback.execute(plan)
```

## 主要错误

- `EvaluationMatrixError`：评测矩阵非法或无法实例化；
- `DeploymentGateError`：部署前门禁失败；
- `RobotUnavailableError`：机器人离线、被占用或校准过期；
- `SafetyViolationError`：检测到安全事件；
- `FeedbackEvidenceError`：失败归因缺少必要 Episode 或日志；
- `RegenerationPlanError`：计划无法满足预算或覆盖约束。
