# 数据管理与模型训练

本模块把已验证 Episode 转换为不可变数据集，完成质量检查、切分和快照，并在项目统一环境的受控子进程中训练策略、VLA 或世界模型。

## 完成目标

```text
Verified Episode → 质量检查 → 训练格式转换 → 数据集快照
→ 训练计划 → 隔离训练 → 权重重载 → 模型登记
```

## 1. 转换 Episode

```python
normalized = px.datasets.ingest(
    verified_episode_bundle_artifact,
    method="<installed-dataset-ingest-method>",
)

dataset_bundle = next(output.artifact for output in normalized.outputs if output.name == "dataset")
```

`verified_episode_bundle_artifact` 是通过执行证据验收的 `EpisodeBundle/1`。转换器必须处理全部模态、动作语义和时间戳，并通过重读验证；训练脚本不得直接遍历运行临时目录。

推荐以 LeRobot v3 作为训练主格式、RLDS 作为外部 Episode 交换格式、rosbag2 + MCAP 保存真机原始流。格式适配器只创建新的训练视图，不改写原始 Episode。

## 2. 质量检查与切分

数据集发布前至少检查：

- 制品标识、字节数和媒体类型完整；
- Episode ID、控制和传感器时间戳单调；
- 观测/动作的 dtype、shape、单位、frame 和频率稳定；
- 成功、终止、失败原因和最后一步语义一致；
- 模态缺失、重复样本、异常值和分布漂移在预算内；
- train/validation/test 按资产、场景和采集批次无交叉泄漏。

切分不是随机行拆分。属于同一真实对象、场景变体或采集批次的 Episode 应进入同一侧，避免模型记住外观或动力学参数。

## 3. 冻结数据集快照

```python
snapshot = px.datasets.snapshot(
    dataset_bundle,
    method="dvc",
)
```

`DatasetSnapshot` 固定 Episode 清单、切分、数据转换方法、质量报告和制品引用。任何新增、删除或重标注都发布新版本，并保留父版本关系。

数据卡应记录用途与禁用范围、真实/仿真/世界模型数据比例、资产和失败分布、模态频率、随机化范围、质量结果、许可证、隐私和已知偏差。

## 4. 选择训练方法

| 目标 | 方法方向 |
| --- | --- |
| 低成本模仿学习 | LeRobot ACT、robomimic BC |
| 多峰接触动作 | Diffusion Policy |
| 离散意图与连续动作 | VQ-BeT |
| 强化学习 | RSL-RL PPO、Stable-Baselines3 |
| VLA | OpenVLA、OpenPI 等独立适配器 |
| 特征或状态世界模型 | DINO-WM、DreamerV3、TD-MPC2 |
| 动作条件未来视频 | IRASim |

方法未出现在 `px.methods.list(...)` 中时不能提交。代码许可证、模型权重许可证和训练数据条款分别登记；核心 Python 环境不直接安装算法依赖。

## 5. 固定训练计划

训练前必须冻结：

- 数据集和 train/validation/test Episode 记录标识；
- 观测/动作 Schema、归一化、时间窗口和 horizon；
- 代码源树、环境锁文件、基础权重和 tokenizer；
- 方法、超参数、随机种子、设备和资源预算；
- 真实、仿真及世界模型生成数据的来源比例。

策略模型与世界模型使用不同模型契约，不能只靠名称区分。预训练权重先进入制品库，再通过
`artifact_id` 绑定；禁止使用 `latest`、可变 URL 或未登记的外部路径。

## 6. 训练并登记模型

```python
from physrealx import MethodCapability, TrainingSpec

trainer = px.methods.bind(
    MethodCapability.POLICY_TRAIN,
    "torch_bc",
    config={
        "observation_signals": ["drawer.position", "drawer.velocity"],
        "epochs": 80,
        "hidden_dim": 32,
        "learning_rate": 0.01,
        "device": "cpu",
        "seed": 17,
    },
)

model_ref = px.training.submit(
    TrainingSpec(
        resource_id="drawer.bc",
        version="1.0.0",
        dataset=dataset_ref,
        trainer=trainer,
    )
)
model = px.training.get_model(model_ref)
```

训练完成后必须从登记权重重新加载并执行推理。训练 loss、梯度和指标需要有限；仅保存进程内模型或最后一个可变 checkpoint 不构成模型发布。

## 实际 cabinet2 训练基线

```bash
python experiments/run_cabinet2_training.py \
  --source /data/ArtModeling/cabinet2 \
  --output-root /data/runs/cabinet2-learning \
  --artgs-run /data/runs/cabinet2-artgs
```

固定运行实际完成：

- 96 个训练 Episode、7,680 个 LeRobot 帧；
- LeRobot ACT 600 次反向传播，loss 从 9.971 降至 0.031；
- 动作条件状态世界模型 71 个 epoch，独立验证归一化 MSE 为 `7.66e-5`；
- 权重重新加载并用于闭环推理。

![策略与世界模型训练](../assets/images/experiments/cabinet2-training-dashboard.png)

数据快照、权重、日志和报告保存在 [`experiments/results/cabinet2-training/`](https://github.com/bupt-yxp/PhysReal-X/tree/main/experiments/results/cabinet2-training)。该实验采用 `engineering_prior` MuJoCo 抽屉模型，证明数据转换和训练实际执行，不作为 cabinet2 真实物性或真机策略结论。

## 输出与验收

- 数据快照可完整重读，切分无来源泄漏；
- 数据、代码、环境、方法、配置和种子可追溯；
- 权重可重新加载并产生符合 Schema 的输出；
- validation/test 不参与训练或定向补数；
- 策略报告训练稳定性、推理延迟和安全相关统计；
- 世界模型报告多步状态误差、漂移和动作敏感性，而非只报告训练 loss。

下一步进入[评测与真机反馈闭环](evaluation-and-feedback.md)。
