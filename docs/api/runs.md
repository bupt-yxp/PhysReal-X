# 运行与数据生成 API

运行服务编译场景、任务、仿真后端和策略方法，逐 Episode 执行并将合格结果原子发布为数据集。
核心包不携带仿真、渲染或 Episode 转换的占位实现；本页中的方法参数必须指向
当前工作区已安装且探针通过的真实适配器。

## `runs.submit`

```python
px.runs.submit(spec: RunSpec) -> RunRecord
```

`RunSpec.backend` 必须是 `simulation.run` 能力的 `MethodSpec`，`RunSpec.policy` 必须是
`policy.inference` 能力的 `MethodSpec`。方法配置保存在各自 `MethodSpec.config` 中，不维护第二套
`backend_config` / `policy_config` 字段。

```python
from physrealx import MethodCapability, RunSpec

backend = px.methods.bind(
    MethodCapability.SIMULATION_RUN,
    "<installed-simulation-method>",
    config={"physics_dt": 0.002},
)
policy = px.methods.bind(
    MethodCapability.POLICY_INFERENCE,
    "motion_planner",
)

record = px.runs.submit(
    RunSpec(
        scene=scene_ref,
        task=task_ref,
        backend=backend,
        policy=policy,
        episode_count=1000,
        base_seed=2026,
        dataset_id="drawer.rollouts",
        dataset_version="1.0.0",
        record_signals=("camera.front.rgb", "drawer.position"),
    )
)
```

提交前平台解析场景和任务版本，核对方法描述符、检查点、观测信号与后端能力；运行 ID 相同且规格
相同时返回原成功记录，不允许以相同身份改写任务。

## `runs.get` / `runs.wait`

```python
px.runs.get(run: str | RunRecord) -> RunRecord
px.runs.wait(run: str | RunRecord) -> RunResult
```

当前执行器同步完成本地运行；`wait` 读取终态结果，非终态会明确失败，不以轮询或解析日志推断状态。

## `runs.episodes`

```python
px.runs.episodes(run: str | RunRecord) -> tuple[Episode, ...]
```

平台从制品存储读取 Episode，重新验证清单引用及全部领域约束。每条 Episode 保留精确场景、任务、
后端与策略方法、种子、时间线、信号 Schema、动作、结果和失败原因。

## 物理轨迹与离线渲染

`runs.submit` 用于策略在环仿真闭环；需要先生成物理状态轨迹、再离线生成多模态时，使用：

```python
simulation = px.simulations.run(
    scene_ref,
    task_ref,
    method="<installed-simulation-method>",
)
trajectory = simulation.outputs[0].artifact

render = px.renders.generate(
    trajectory,
    method="<installed-render-method>",
)
events = px.events.synthesize(
    render.outputs[0].artifact,
    method="<installed-event-method>",
)
points = px.pointclouds.project(
    depth_artifact,
    method="<installed-point-cloud-method>",
)
episode = px.episodes.pack(
    trajectory,
    method="<installed-episode-packer>",
)
```

| API | 输出 |
| --- | --- |
| `simulations.run` | `state_trajectory` + `simulation_report` |
| `renders.generate` | `render_streams` + `camera_metadata` |
| `events.synthesize` | `event_stream` |
| `pointclouds.project` | `point_cloud` |
| `episodes.pack` | `episode_bundle` + `episode_manifest` |

渲染方法必须消费 `state_trajectory`，不得重新执行物理。时间戳、相机参数和轨迹记录标识必须进入
模态清单，否则 RGB、深度、接触和关节状态不具备同步证据。
上述尖括号是部署方法名的占位说明，不是平台内置默认值。调用前应通过
`px.methods.installations(...)` 查询实际部署，并校验输入资源的版本、Schema 与依赖闭包。

## 外部执行证据闭包

`ExecutionLifecycle` 不执行仿真器、渲染器或 Episode packer；它只校验已注册方法返回的真实制品
证据。`StateTrajectoryEvidence` 与 `RenderStreamEvidence` 必须逐帧匹配 index、纳秒时间戳和
state-frame 记录标识，packer 还必须绑定同一轨迹、渲染结果、场景、任务、方法和种子。通过后
先得到 `VerifiedEpisodeBundle`，实际物化并校验引用后再聚合为 `VerifiedEpisodeCollection`
(`physrealx.verified-episode-collection/1`, `FINALIZED`)；它是数据集就绪门禁唯一接受的
`episodes` 上游契约。测试夹具中的方法名不表示仓库内置了相应方法。

## 发布语义

1. 每条 Episode 先写入标识符寻址制品，再写稳定记录标识；
2. 信号集合、dtype、shape、单位和 frame 在 Episode 内必须稳定；
3. 全部 Episode 完成后构建 `DatasetManifest` 并原子登记数据集引用；
4. 失败运行保存稳定错误码，不发布不完整数据集；
5. session、策略和临时资源在成功或异常路径均关闭。

## 主要错误

- `ResourceNotFoundError`：场景、任务、资产或方法不存在；
- `ResourceConflictError`：制品引用、方法身份或运行身份冲突；
- `DomainValidationError`：任务信号、能力、检查点或 Episode 语义不合法；
- `BackendExecutionError`：后端 reset/step/close 失败。
