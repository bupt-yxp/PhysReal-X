# 训练 API

训练服务消费不可变数据集，在已注册训练方法中执行，并发布带完整来源的模型清单。大型训练框架应
运行在隔离 worker 中，不能成为核心包的导入依赖。

## `training.submit`

```python
px.training.submit(spec: TrainingSpec) -> VersionedRef
```

```python
from physrealx import MethodCapability, TrainingSpec

trainer = px.methods.bind(
    MethodCapability.POLICY_TRAIN,
    "torch_bc",
    config={
        "observation_signals": ["drawer.position", "drawer.velocity"],
        "epochs": 80,
        "learning_rate": 0.01,
        "device": "cpu",
        "seed": 17,
    },
)

model = px.training.submit(
    TrainingSpec(
        resource_id="drawer.bc",
        version="1.0.0",
        dataset=dataset_ref,
        trainer=trainer,
    )
)
```

提交时平台核对数据集摘要、训练方法完整描述符与可选检查点，再将 Episode 流交给训练适配器。
训练器不得改变输入动作 Schema，权重必须先进入制品存储，模型清单通过校验后才原子发布。

## `training.get_model`

```python
px.training.get_model(model: VersionedRef) -> ModelManifest
```

`ModelManifest` 保存训练方法与推理方法 `MethodSpec`、数据集精确引用、权重摘要、观测/动作
Schema、规范化训练配置和有限指标。推理方法必须已在当前工作区注册。

## `training.list_models`

```python
px.training.list_models() -> list[VersionedRef]
```

## 外部模型登记与世界模型 rollout

```python
registration = px.models.register(model_ref, method="mlflow")
rollout = px.world_models.rollout(
    world_model_ref,
    context_artifact,
    action_sequence,
    method="dino_wm",
)
```

`models.register` 产生与数据、代码、环境、配置、指标和权重摘要绑定的登记证据；
`world_models.rollout` 产生 `rollout` 和 `uncertainty` 制品。方法只能返回已声明的特征、
视频、状态或价值输出，平台不伪造缺失模态。

## 方法选择

| 目标 | 能力方法 |
| --- | --- |
| 可执行参考闭环 | `torch_bc` |
| 通用示范学习 | `lerobot_act`、`robomimic_bc` |
| 多峰连续控制 | `diffusion_policy` |
| 离散意图与连续动作 | `vqbet` |
| GPU 强化学习 | `rsl_rl_ppo` |
| 动作条件世界模型 | `dino_wm`、`irasim`、`dreamerv3`、`tdmpc2` |

外部方法通过独立扩展包注册；是否可用以 `px.methods.list(...)` 为准。方法名称不能替代版本、源码、
镜像、许可证、权重和配置证据。

## 主要错误

- `ResourceNotFoundError`：数据集、训练方法或推理方法未注册；
- `ResourceConflictError`：数据摘要、方法描述符或输出模型身份冲突；
- `DomainValidationError`：信号 Schema、检查点、权重或模型清单不符合契约。
