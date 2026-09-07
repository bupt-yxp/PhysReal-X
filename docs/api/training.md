# 训练与世界模型 API

## 策略

`training.submit(TrainingSpec) -> VersionedRef` 验证冻结数据、训练方法与权重，实际训练后发布 `ModelManifest`。`training.get_model(ref)` / `training.list_models()` 读取已登记模型。

```python
from physrealx import MethodCapability, TrainingSpec

model_ref = px.training.submit(
    TrainingSpec(
        resource_id="cabinet2.act",
        version="1.0.0",
        dataset=train_dataset,
        trainer=px.methods.bind(
            MethodCapability.POLICY_TRAIN,
            "act",
            config={"steps": 300, "device": "cuda", "seed": 2026},
        ),
    )
)
```

`act` 执行官方 LeRobot CVAE/Transformer 网络；默认观察 camera.front.rgb、joint.position、joint.velocity，配置在 `ACTTrainConfig` 中验证。推理使用保存的权重、训练归一化与完整动作块。`torch_bc` 作为数值契约基线独立保留，不能冒充 ACT。

## 世界模型

```python
trained = px.world_models.train(
    snapshot_ref,
    encoder_ref,
    method="dino_wm",
    config={
        "device": "cuda",
        "history": 1,
        "steps": 1800,
        "training_horizon": 16,
        "evaluation_horizon": 100,
    },
)
model = next(item.artifact for item in trained.outputs if item.name == "model")
prediction = px.world_models.rollout(
    model, context_ref, actions_ref, method="dino_wm", config={"device": "cuda"}
)
```

训练输出 `WorldModelBundle` 与 `TrainingReport`，沿核心 `TrainingLifecycle` 固定快照、代码、环境、编码器和实际 checkpoint。模型包与 ACT 的 `ModelManifest` 按用途区分，不重复定义数据快照。

上下文为 `WorldModelContext(episode=EpisodeSummary, start_index=...)`；动作是 `WorldModelActions(values=..., interval_seconds=...)`。上下文只读取起点之前必要帧，检查动作维数、历史完整性、单位及频率，最多生成 128 步未来 RGB 与状态。输出名为 `rollout` 和 `uncertainty`；后者明确标记未校准。

方法依赖按需安装到现有环境，权重不可变发布，训练结束后必须重载推理。配置和适配详情见[训练指南](../user-guide/data-management-and-training.md)。
