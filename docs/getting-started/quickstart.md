# 快速开始

本页用一个“打开抽屉”任务展示完整主链路：注册资产与场景、编译任务、生成数据集、训练策略并进行独立评测。

## 1. 连接平台

```python
from physrealx import PhysRealX

px = PhysRealX.connect("local:///.physrealx")
```

`local:///` 表示连接本机端点，HTTPS 地址表示连接远程控制服务。端点决定控制服务的位置；仿真与训练由平台根据后端能力调度到已注册工作节点。

## 2. 注册资产

```python
from physrealx import AssetSpec

drawer = px.assets.register(AssetSpec.from_file("assets/drawer.asset.yaml"))
robot = px.assets.register(AssetSpec.from_file("assets/franka.asset.yaml"))
```

资产包含视觉几何、碰撞几何、link/joint 结构、物理参数、坐标系和制品摘要。注册时会执行结构与物理一致性校验。

## 3. 组合场景并编译任务

```python
from physrealx import SceneSpec, TaskDraft

scene = px.scenes.register(SceneSpec.from_file("scenes/drawer_table.scene.yaml"))

task = px.tasks.compile(
    TaskDraft(
        instruction="打开抽屉至少 30 厘米",
        scene=scene,
        robot="franka",
        target="drawer.drawer_link",
    )
)
```

任务编译会把自然语言草案转换为可执行条件：观测模式、动作空间、初始状态、成功条件、终止条件和评测指标。

## 4. 生成多模态数据

```python
from physrealx import RunSpec

run = px.runs.submit(
    RunSpec(
        scene=scene,
        task=task,
        backend="isaac_lab",
        policy="motion_planner",
        episodes=1_000,
        seed=2026,
        record=(
            "camera.front.rgb",
            "camera.front.depth",
            "robot.joint_position",
            "drawer.joint_position",
            "contact.force",
        ),
    )
)

result = px.runs.wait(run)
dataset = px.datasets.get(result.dataset)
```

平台只在 1,000 条 Episode 全部完成、信号模式一致且制品通过摘要校验后发布数据集。

## 5. 训练策略

```python
from physrealx import TrainingSpec

training = px.training.submit(
    TrainingSpec(
        dataset=dataset.ref,
        trainer="diffusion_policy",
        model_id="drawer.diffusion-policy",
        seed=17,
        config={"epochs": 200, "device": "cuda"},
    )
)

model = px.training.wait(training).model
```

模型清单记录数据集摘要、训练器版本、配置、随机种子、权重摘要、输入输出模式和指标。

## 6. 独立评测

```python
from physrealx import EvaluationSpec

evaluation = px.evaluations.submit(
    EvaluationSpec(
        model=model,
        matrix="drawer.heldout-v1",
        episodes=200,
        seed=100_000,
    )
)

report = px.evaluations.wait(evaluation).report
print(report.success_rate)
print(report.failures.by_category())
```

评测种子、场景矩阵和生成数据与训练阶段独立。失败案例会按感知、规划、控制、接触、超时和数据质量分类。

## 命令行等价流程

```bash
physrealx assets register assets/drawer.asset.yaml
physrealx assets register assets/franka.asset.yaml
physrealx scenes register scenes/drawer_table.scene.yaml
physrealx tasks compile tasks/open_drawer.task.yaml
physrealx runs submit runs/open_drawer.run.yaml --wait
physrealx training submit training/diffusion_policy.yaml --wait
physrealx evaluations submit evaluation/heldout.yaml --wait
```

接下来阅读[第一个完整项目](first-project.md)，了解推荐目录、规格文件和验收方法。
