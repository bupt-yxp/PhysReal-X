# 扩展协议

平台通过窄协议接入外部能力。插件以 Python entry point 或隔离工作节点注册；公共协议只使用领域模型，不泄漏仿真器、Torch、ROS 2 或存储对象。

## 插件描述

```python
PluginDescriptor(
    kind: str,
    name: str,
    version: str,
    api_version: str,
    capabilities: Mapping[str, JSONValue],
)
```

插件身份由 `kind + name + version + api_version` 确定。插件代码或容器变化必须更新版本或构建摘要。

## SimulationBackend

```python
class SimulationBackend(Protocol):
    descriptor: PluginDescriptor

    def capabilities(self) -> BackendCapabilities: ...

    def create_session(
        self,
        scene: SceneSpec,
        task: TaskSpec,
        config: Mapping[str, JSONValue],
    ) -> SimulationSession: ...
```

## SimulationSession

```python
class SimulationSession(Protocol):
    @property
    def clock(self) -> ClockSpec: ...

    def reset(self, seed: int) -> Frame: ...
    def step(self, action: DataValue) -> StepResult: ...
    def snapshot(self) -> ArtifactRef: ...
    def close(self) -> None: ...
```

会话方法不得发布数据集或修改资源目录；它只执行世界并返回标准帧。

## Policy

```python
class Policy(Protocol):
    descriptor: PluginDescriptor

    def reset(self, seed: int, context: PolicyContext) -> None: ...
    def act(self, frame: Frame, task: TaskSpec, step_index: int) -> DataValue: ...
    def close(self) -> None: ...
```

策略可以是脚本、规划器、遥操作、学习模型或世界模型规划器。

## Trainer

```python
class Trainer(Protocol):
    descriptor: PluginDescriptor

    def validate(self, spec: TrainingSpec, schema: DatasetSchema) -> ValidationReport: ...

    def train(
        self,
        spec: TrainingSpec,
        data: DatasetReader | EnvironmentFactory,
        context: TrainingContext,
    ) -> ModelManifest: ...
```

训练器通过 `TrainingContext` 写入指标、检查点和制品，不直接访问控制面数据库。

## AssetBuilder / AssetImporter

```python
class AssetBuilder(Protocol):
    descriptor: PluginDescriptor

    def build(
        self,
        source: CaptureRef | DatasetRef,
        config: Mapping[str, JSONValue],
        context: BuildContext,
    ) -> AssetSpec: ...
```

导入器是专门的 builder，用于 USD、URDF、MJCF、CAD、Mesh 和 Gaussian。

## DatasetExporter

```python
class DatasetExporter(Protocol):
    descriptor: PluginDescriptor

    def export(
        self,
        manifest: DatasetManifest,
        reader: DatasetReader,
        destination: str,
        config: Mapping[str, JSONValue],
    ) -> ExportManifest: ...
```

## RobotBackend

```python
class RobotBackend(Protocol):
    descriptor: PluginDescriptor

    def capabilities(self) -> RobotCapabilities: ...
    def deploy(self, spec: DeploymentSpec) -> RobotSession: ...
```

`RobotSession` 必须实现状态、观测、动作、安全停止、Episode 记录和幂等关闭。

## FeedbackAnalyzer

```python
class FeedbackAnalyzer(Protocol):
    descriptor: PluginDescriptor

    def analyze(
        self,
        episodes: Iterable[Episode],
        context: FeedbackContext,
    ) -> FailureAnalysis: ...
```

## 插件注册

```toml
[project.entry-points."physrealx.plugins"]
isaac_lab = "physrealx_isaac:register"
```

```python
def register(registry: PluginRegistry) -> None:
    registry.register_backend(IsaacLabBackend())
```

隔离工作节点使用相同 `PluginDescriptor` 和协议语义，通过远程传输适配层调用。

## 验收要求

- 固定 API 和插件版本；
- 能力声明准确，运行前能快速拒绝不支持请求；
- create/reset/step/close 或对应生命周期完整且清理幂等；
- 信号 dtype、shape、unit、frame 与时间戳严格一致；
- 固定种子满足声明的复现等级；
- 取消、超时、节点失联和制品损坏有结构化错误；
- 不发布部分清单，不吞掉后端异常；
- 通过共享契约测试和真实运行时集成测试。
