# 训练与世界模型 API

## 策略

`training.submit(TrainingSpec) -> VersionedRef` 验证冻结数据、训练方法与权重，实际训练后发布 `ModelManifest`。`training.get_model(ref)` / `training.list_models()` 读取已登记模型。

```python
from physrealx import MethodCapability, TrainingSpec

model_ref = px.training.submit(
    TrainingSpec(
        resource_id="oven4.act",
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

`px.world_models.prepare_encoder(cache=..., weights_path=...)` 准备当前支持的 DINOv2 ViT-S 编码器。默认校验并复用工作区缓存；可显式传入本地权重，否则从官方地址下载。下载大小、权重结构与有限值由项目检查，缓存仅在导入成功后替换；Notebook 无需实现网络与缓存逻辑。

```python
trained = px.world_models.train(
    snapshot_ref,
    encoder_ref,
    method="dino_wm",
    config={
        "device": "cuda",
        "history": 1,
        "steps": 1800,
        "training_horizon": 100,
        "evaluation_horizon": 100,
    },
)
model = trained.output("model")
prediction = px.world_models.rollout(
    model, context_ref, actions_ref, method="dino_wm", config={"device": "cuda"}
)
```

训练输出 `WorldModelBundle` 与 `TrainingReport`，沿核心 `TrainingLifecycle` 固定快照、代码、环境、编码器和实际 checkpoint。模型包与 ACT 的 `ModelManifest` 按用途区分，不重复定义数据快照。

上下文为 `WorldModelContext(episode=EpisodeSummary, start_index=...)`；动作是 `WorldModelActions(values=..., interval_seconds=...)`。上下文只读取起点之前必要帧，检查动作维数、历史完整性、单位及频率，最多生成 128 步未来 RGB 与状态。输出名为 `rollout` 和 `uncertainty`；后者明确标记未校准。

`world_models.evaluate_rollout(rollout, reference_episode, start_index=...)` 是固定评估服务，不选择 method。它严格核对预测与参考 episode 的时间轴、信号维数、单位和坐标系，逐状态信号报告 RMSE、相同起点 persistence RMSE、误差比及 `BEATS_PERSISTENCE` 状态。不同单位不合并成一个分数，也不使用与对象或预测时长无关的固定误差魔数。

方法依赖按需安装到现有环境，权重不可变发布，训练结束后必须重载推理。配置和适配详情见[训练指南](../user-guide/data-management-and-training.md)。

世界模型训练展开长度支持 1–128 步；示例在 20 Hz 下使用 100 步覆盖五秒验收时长。检查点按验证集的状态 RMSE 比率平方、潜在特征 MSE 比率和 RGB MSE 比率三者最大值选择（均以保持原状态为基线），避免量纲或单项误差主导选择；各项比率保存在训练诊断中，测试集仅用于最终评估。
