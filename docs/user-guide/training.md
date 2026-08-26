# 策略训练

训练服务消费版本化数据集或在线环境，输出带完整来源与版本记录的模型清单。训练器负责批处理、设备和分布式执行；平台负责输入校验、任务生命周期、制品发布和结果登记。

## 支持的训练范式

| 范式 | 输入 | 产出 |
| --- | --- | --- |
| 行为克隆 / Diffusion Policy | 观测—动作 Episode | 连续控制策略 |
| 强化学习 | 向量化环境或离线数据 | 策略、价值网络和训练统计 |
| VLA 微调 | 图像、语言、动作和任务标签 | 视觉—语言—动作模型 |
| 世界模型 | 状态/观测、动作、下一状态 | 动作条件动力学模型 |
| 混合训练 | 真实 + 仿真 + 反事实数据 | 带来源加权的统一模型 |

## 提交训练

```python
training = px.training.submit(
    TrainingSpec(
        resource_id="model.drawer-dp",
        version="1.0.0",
        trainer="diffusion_policy",
        dataset="dataset.drawer-hybrid@1.0.0",
        split="train",
        seed=17,
        config={
            "epochs": 200,
            "batch_size": 256,
            "learning_rate": 1e-4,
            "device": "cuda",
            "precision": "bf16",
        },
        validation={"split": "validation", "every_steps": 1000},
    )
)
```

平台在调度前检查训练器能否消费数据集的观测和动作模式，以及请求设备、精度和分布式能力是否可用。

## 模仿学习

模仿学习建议显式定义：

- 历史观测窗口和动作预测 horizon；
- 图像预处理和数据增强；
- 成功/失败/恢复样本权重；
- 动作归一化、坐标 frame 和控制频率；
- 模型推理所需信号子集。

任何影响模型输入语义的转换都必须版本化，不能只存在于训练脚本的匿名 lambda 中。

## 强化学习

在线 RL 的 `TrainingSpec` 引用 `EnvironmentSpec`：

```python
training = px.training.submit(
    TrainingSpec(
        trainer="ppo",
        environment={
            "scene": "scene.drawer-table@1.0.0",
            "task": "task.open-drawer@1.0.0",
            "backend": "isaac_lab",
            "num_envs": 4096,
        },
        seed=17,
        config={"steps": 100_000_000},
    )
)
```

环境规格固定观测、动作、奖励、终止、事件、课程和随机化。训练日志还应记录每秒环境步、成功率、回报、熵、值损失和失败分布。

## VLA 训练

VLA 数据视图需要对齐：

```text
视觉历史 + 语言指令 + 机器人本体状态 → 动作 chunk
```

必须固定：tokenizer、图像预处理、语言模板、动作 tokenizer/归一化、机器人 embodiment 映射、时间窗口与丢帧策略。

## 世界模型

世界模型训练数据应显式区分：

- 可观测图像/点云；
- 特权状态；
- 动作；
- 接触与物理参数；
- 真实、仿真和模型生成来源。

模型生成的反事实数据不能与真实数据混淆，其每条 Episode 必须记录父观测、候选动作、模型版本和采样配置。

## 监控与断点

```python
for event in px.training.watch(training):
    print(event.step, event.metrics, event.resources)
```

检查点是内容寻址制品，记录全局步、优化器、调度器、随机状态和数据游标。恢复训练必须引用精确检查点，不使用“latest”作为最终可追溯依据。

## 模型清单

训练完成后 `ModelManifest` 包含：

- 模型 ID、版本、架构和权重摘要；
- 训练器名称、版本和容器/环境摘要；
- 数据集与 split 摘要；
- 输入观测、语言和动作模式；
- 配置、种子、训练时长和资源；
- 最终指标、检查点与训练日志；
- 推理适配器和兼容机器人/任务。

## 训练与评测分离

训练过程中的 validation 用于模型选择，不能替代独立评测。最终结果必须在冻结评测矩阵、独立种子和未见资产/场景上由评测服务产生。
