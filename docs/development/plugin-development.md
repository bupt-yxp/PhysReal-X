# 插件开发

本页以仿真后端为例说明开发流程。其他插件遵循相同原则：先声明能力，再实现窄协议，最后通过共享契约测试。

## 创建独立包

```text
physrealx-isaac/
├── pyproject.toml
├── src/physrealx_isaac/
│   ├── __init__.py
│   ├── backend.py
│   ├── session.py
│   ├── conversion.py
│   └── worker.py
└── tests/
    ├── contract/
    └── integration/
```

重型后端使用独立环境和 worker 进程；核心测试可以用轻量 fake 实现协议。

## 实现描述与能力

```python
class IsaacLabBackend:
    descriptor = PluginDescriptor(
        kind="simulation_backend",
        name="isaac_lab",
        version="1.0.0",
        api_version="1",
        capabilities={"runtime": "worker"},
    )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            rigid_body=True,
            articulation=True,
            contact_force=True,
            batched=True,
            deterministic_reset=False,
            observation_keys=("rgb", "depth", "joint_state", "contact_force"),
            action_kinds=("joint_position", "ee_delta_pose"),
        )
```

能力声明必须保守。只有共享测试和目标运行时都验证过的能力才能设为 true。

## 转换边界

`conversion.py` 负责：

- `AssetSpec` → USD / 后端实体；
- `SceneSpec` → stage、传感器与随机化；
- `TaskSpec` → 观测、动作和终止读取；
- 后端张量/图像 → `Frame` 与 `DataValue`；
- 平台动作 → 后端控制器命令。

转换函数尽量纯化并单独测试。后端对象不能进入领域模型或应用服务。

## 生命周期实现

```python
class IsaacSession:
    def reset(self, seed: int) -> Frame: ...

    def step(self, action: DataValue) -> StepResult: ...

    def snapshot(self) -> ArtifactRef: ...

    def close(self) -> None:
        if self._closed:
            return
        ...
        self._closed = True
```

`close()` 必须幂等，并在中途异常时释放 GPU、stage、传感器、临时文件和 IPC 句柄。

## 注册

```toml
[project.entry-points."physrealx.plugins"]
isaac_lab = "physrealx_isaac:register"
```

```python
def register(registry: PluginRegistry) -> None:
    registry.register_backend(IsaacLabBackend())
```

## 契约测试

```python
def test_backend_contract(backend_contract):
    backend_contract.verify(IsaacLabBackend())
```

共享套件覆盖：

- 描述、版本与能力；
- 场景创建、reset、step、snapshot、close；
- dtype、shape、unit、frame 和时间戳；
- 动作边界；
- 固定种子复现等级；
- 缺失能力快速失败；
- 取消、超时、异常和重复关闭；
- 大制品写入与摘要；
- 多 Episode 会话复用。

## 工作节点

worker 负责运行时隔离、设备管理、心跳、租约和上传，不实现业务状态机。任务消息只包含已解析规格、制品引用、输出上传凭据和租约信息。

worker 必须：

- 使用短期凭据；
- 定期续租和报告进度；
- 租约失效后停止发布；
- 对相同 Episode 尝试保持幂等；
- 上传完成后由控制面决定是否进入正式清单。

## 发布门槛

插件只有在以下条件均满足时发布：

1. 核心契约测试通过；
2. 目标后端真实安装环境集成测试通过；
3. 至少一个端到端任务通过；
4. 性能、显存和稳定性基准完成；
5. 已知限制与能力声明一致；
6. 文档包含安装、示例、故障排查和版本矩阵。
