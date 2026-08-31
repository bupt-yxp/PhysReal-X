# 第一个可复现契约实验

本教程把规格代码、工作区和大制品分开，建立一个可重复执行的项目。
示例的 Reference backend 和 `torch_bc` 只完整覆盖平台契约，不会产生真实物理、
已恢复关节、已辨识物性或真机部署证据。

## 推荐目录

```text
open-drawer/
├── pyproject.toml
├── workflow.py
├── specs/
│   ├── assets/
│   ├── scenes/
│   ├── tasks/
│   └── recipes/
├── tests/
└── .physrealx/        # 不提交 Git
```

规格和测试进入 Git；RGB-D、Mesh、Episode 和权重进入制品存储，通过平台生成的制品标识引用。

## 工作流

```python
from physrealx import MethodCapability, PhysRealX, TrainingSpec
from physrealx.reference import (
    reference_asset,
    reference_run,
    reference_scene,
    reference_task,
)

px = PhysRealX.open(".physrealx")

asset = reference_asset()
scene = reference_scene(asset)
task = reference_task(scene)
asset_ref = px.assets.register(asset)
scene_ref = px.scenes.register(scene)
task_ref = px.tasks.register(task)

run = px.runs.submit(reference_run(scene, task, episode_count=16, base_seed=2026))
result = px.runs.wait(run)
dataset = px.datasets.get(result.dataset)

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

evaluation = px.evaluations.submit(
    resource_id="drawer.bc.heldout",
    version="1.0.0",
    model=model,
    run=reference_run(scene, task),
    episode_count=16,
    base_seed=100_000,
)

print(asset_ref, scene_ref, task_ref)
print(len(dataset.episodes), px.evaluations.get_report(evaluation).success_rate)
```

运行：

```bash
python workflow.py
```

## 替换为真实项目

1. 用 `assets.inspect_source` 和 `assets.ingest` 导入已采集 ArtModeling 数据；
2. 安装数字孪生扩展，并以逐阶段 `MethodSpec` 构建资产；
3. 使用 `SceneSpec` 与 `TaskSpec` 显式定义实体、信号、动作和成功条件；
4. 从 `px.methods.list(...)` 选择仿真、策略和训练方法，不在业务代码导入后端对象；
5. 使用独立资产/场景/种子构建评测集，并保留失败 Episode。

## 验收清单

- 所有资源引用均带制品标识，同一版本不能覆盖；
- 方法证据包含源码 revision、许可证、环境、权重和配置；
- 资产视觉/碰撞、关节、物性和回放均通过质量门；
- Episode 时间戳、信号 Schema、末帧和失败语义完整；
- 数据集中的每条 Episode 可从制品存储重新加载并验证；
- 训练与评测使用不同种子，模型可回溯到数据与方法；
- 失败结果保留证据，不以截图或单一平均分替代验收。
