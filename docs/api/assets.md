# 资产 API

资产服务统一处理已有文件导入、真实采集构建、结构化规格注册、物理校准和质量验证。

## `assets.register`

```python
px.assets.register(
    spec: AssetSpec,
    *,
    idempotency_key: str | None = None,
) -> ResolvedRef[AssetSpec]
```

规范化序列化 `AssetSpec`、验证外部制品并发布不可变资产版本。同一 `resource_id@version` 若内容不同，抛出 `ResourceConflictError`。

## `assets.import_asset`

```python
px.assets.import_asset(
    source: str | Path | ArtifactRef,
    *,
    importer: str,
    resource_id: str | None = None,
    version: str = "1.0.0",
    profile: str = "auto",
    config: Mapping[str, JSONValue] | None = None,
) -> JobRef[ResolvedRef[AssetSpec]]
```

支持 USD、URDF、MJCF、Mesh、CAD 和标准 Gaussian 适配器。返回异步任务，因为导入可能包含上传、转换、碰撞生成和验证。

## `assets.build`

```python
px.assets.build(
    source: CaptureRef | DatasetRef,
    *,
    pipeline: str,
    resource_id: str,
    version: str,
    config: Mapping[str, JSONValue] | None = None,
) -> JobRef[ResolvedRef[AssetSpec]]
```

从真实采集构建数字孪生。`pipeline` 可编排深度补全、几何重建、部件分解、关节估计、碰撞生成和物理测量适配器。

## `assets.get`

```python
px.assets.get(ref: AssetLike) -> AssetSpec
```

若传入可读引用，返回目录中对应精确版本；若传入带摘要引用，同时校验摘要。

## `assets.list`

```python
px.assets.list(
    *,
    category: str | None = None,
    tags: Sequence[str] = (),
    text: str | None = None,
    limit: int = 100,
    cursor: str | None = None,
) -> Page[AssetSummary]
```

## `assets.validate`

```python
px.assets.validate(
    asset: AssetSpec | AssetLike,
    *,
    profile: Literal["schema", "simulation", "production"] = "production",
    backend: str | None = None,
) -> ValidationReport
```

| Profile | 检查 |
| --- | --- |
| `schema` | 字段、标识、单位、坐标系、图结构和参数范围 |
| `simulation` | 外部制品、碰撞、惯量、静态稳定与全关节运动 |
| `production` | 额外执行真实重播、性能预算和指定后端兼容性 |

## `assets.calibrate`

```python
px.assets.calibrate(
    asset: AssetLike,
    *,
    observations: DatasetLike,
    parameters: Sequence[str],
    objective: Sequence[str],
    backend: str,
    config: Mapping[str, JSONValue] | None = None,
) -> JobRef[ResolvedRef[AssetSpec]]
```

校准始终发布新资产版本，不修改输入资产。

## `assets.diff`

```python
px.assets.diff(left: AssetLike, right: AssetLike) -> AssetDiff
```

按语义比较几何制品、link/joint、坐标、碰撞、物理参数、校准和元数据，而不只比较 JSON 文本。

## 示例

```python
job = px.assets.import_asset(
    "drawer.usd",
    importer="usd",
    resource_id="drawer.real-001",
    profile="articulated_object",
)
ref = px.jobs.wait(job).result

report = px.assets.validate(ref, profile="production", backend="isaac_sim")
report.raise_for_errors()
```

## 主要错误

- `AssetGraphError`：link/joint 图非法；
- `ArtifactIntegrityError`：外部制品摘要或大小不匹配；
- `CoordinateFrameError`：坐标或单位不一致；
- `BackendCompatibilityError`：指定后端无法物化资产；
- `ResourceConflictError`：不可变版本冲突。
