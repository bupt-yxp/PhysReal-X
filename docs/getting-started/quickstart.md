# 快速开始

本页运行内置“打开抽屉”确定性参考切片，验证资产、场景、任务、运行、
Episode 与数据集的契约链路。该参考后端是回归测试用状态机，不代表 MuJoCo/Isaac Sim
真实物理、`cabinet2` 数字孪生、外部策略训练或 ROS 2 真机部署已经完成。

## 1. 打开工作区

```python
from physrealx import PhysRealX

px = PhysRealX.open(".physrealx-demo")
```

## 2. 注册资产、场景与任务

```python
from physrealx.reference import reference_asset, reference_scene, reference_task

asset = reference_asset()
scene = reference_scene(asset)
task = reference_task(scene)

asset_ref = px.assets.register(asset)
scene_ref = px.scenes.register(scene)
task_ref = px.tasks.register(task)
```

注册过程把场景中的资产和任务中的场景解析为带摘要的精确引用。重复注册相同内容是幂等的，
同一 `resource_id@version` 对应不同内容会被拒绝。

## 3. 生成 Episode 数据集

```python
from physrealx.reference import reference_run

record = px.runs.submit(reference_run(scene, task, episode_count=8, base_seed=7))
result = px.runs.wait(record)
dataset = px.datasets.get(result.dataset)

print(record.status)
print(len(dataset.episodes), dataset.total_steps)
```

运行规格已绑定精确后端和策略 `MethodSpec`。每条 Episode 保存种子、方法描述符、场景/任务摘要、
连续时间线、观测、动作、奖励、终止语义和失败原因；全部 Episode 验证完成后才发布数据集。

## 4. 训练契约参考策略

训练扩展需要 `.[train]` 和 PyTorch 环境。`torch_bc` 只是小型数值回归基线，用于验证
数据集→训练→模型清单的契约，不可作为真实机器人能力结论。

```python
from physrealx import MethodCapability, TrainingSpec

trainer = px.methods.bind(
    MethodCapability.POLICY_TRAIN,
    "torch_bc",
    config={"device": "cpu", "epochs": 80, "seed": 17},
)
model = px.training.submit(
    TrainingSpec(
        resource_id="drawer.bc",
        version="1.0.0",
        dataset=result.dataset,
        trainer=trainer,
    )
)
manifest = px.training.get_model(model)

print(manifest.metrics)
```

模型清单绑定数据集摘要、训练与推理方法、训练配置、权重摘要、信号模式和指标。

## 5. 独立评测

```python
evaluation = px.evaluations.submit(
    resource_id="drawer.bc.heldout",
    version="1.0.0",
    model=model,
    run=reference_run(scene, task),
    episode_count=16,
    base_seed=100_000,
)
report = px.evaluations.get_report(evaluation)

print(report.success_rate, report.failures)
```

评测使用独立种子，并将推理方法切换为模型清单中登记的策略；结果保留评测数据集与失败证据。

## 命令行

```bash
physrealx demo --workspace .physrealx-demo --episodes 8
physrealx demo --workspace .physrealx-train --episodes 8 --train --device cpu
physrealx methods --workspace .physrealx-demo
```

接下来阅读[第一个可复现契约实验](first-project.md)了解目录、规格和质量验收，或进入
[资产与数字孪生](../user-guide/assets.md)处理已采集 ArtModeling 数据。
