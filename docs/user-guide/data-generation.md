# 多模态数据生成

数据生成不是“把仿真截图保存下来”，而是生成具有统一时间语义、完整监督信号、精确来源记录和可验证质量的机器人交互 Episode。

## 轨迹来源

| 来源 | 典型用途 |
| --- | --- |
| 运动规划器 | 生成几何可达、物理可执行的专家轨迹 |
| 遥操作 | 获取人类示范和恢复行为 |
| 脚本策略 | 构造确定性基线与契约测试 |
| 学习策略 | 采集 on-policy 数据和失败分布 |
| 世界模型 rollout | 生成反事实候选和长尾想象数据 |
| 真实机器人 | 提供 Reality Anchor、校准和最终评测 |

所有来源都输出相同 Episode 合同，并通过 `source` 与生成器版本区分。

## 记录哪些信号

### 视觉与几何

- RGB、HDR、Depth、Normal、Optical Flow；
- 语义/实例/部件 Mask；
- 点云、相机内外参和曝光信息；
- 可见性、遮挡率和对象投影。

### 机器人与物体状态

- 关节位置、速度、力矩；
- 末端位姿、夹爪状态、控制目标；
- 物体位姿、关节状态和速度；
- 接触对、法向、冲量、力与触觉。

### 任务与监督

- 语言指令、任务图节点和技能标签；
- 动作、奖励、折扣、成功与终止；
- 事件、失败类别、恢复阶段和安全触发；
- 场景参数与随机化实值。

只记录训练与诊断需要的信号。未使用的高带宽模态会显著降低仿真吞吐并增加存储成本。

## Episode 时间语义

每个普通步骤保存当前观测以及从该观测产生的动作：

```text
step 0: observation₀ + state₀ + action₀ + reward₀
step 1: observation₁ + state₁ + action₁ + reward₁
...
last:   observationₙ + stateₙ
```

末帧没有动作、奖励和折扣。成功 Episode 必须以 terminal 末帧结束；失败 Episode 必须包含失败原因。

## 异步传感器

Frame 的控制时间戳和信号自身时间戳分开保存：

```python
DataValue(
    dtype="float32",
    shape=(6,),
    unit="N",
    frame="wrist",
    timestamp=12.405,
    artifact=force_chunk,
)
```

原始层保留真实采样时间。训练视图可以选择：最近邻、线性插值、时间窗口聚合或事件累积，但转换必须形成带版本的新数据视图。

## 大对象与小状态

- 小型数值状态可内联；
- 图像、点云、长序列和权重写入制品库；
- `DataValue` 保存摘要、大小、媒体类型、shape、dtype、unit、frame 和时间戳；
- 清单不直接嵌入 Base64 或大数组。

## 成功与失败采样

数据计划应显式定义目标分布：

```yaml
sampling:
  success: 0.60
  recoverable_failure: 0.25
  hard_failure: 0.15
```

失败样本应覆盖：

- 遮挡和感知误判；
- 抓取偏移、滑落与碰撞；
- 摩擦、质量、刚度和执行误差；
- 规划不可达和控制振荡；
- 安全终止与超时；
- 恢复动作是否成功。

## 在线质量门

每条 Episode 发布前检查：

1. 信号集合、dtype、shape、unit 和 frame 与任务一致；
2. 控制时间戳严格递增，传感器时间戳单调；
3. 数值不包含 NaN、Inf 或越界动作；
4. 制品摘要、大小和媒体类型正确；
5. 成功、终止、失败原因和最后一步语义一致；
6. 资产、场景、任务、后端、策略和种子引用已解析。

数据集级质量门还包括模态缺失率、类别/任务覆盖、重复率、异常值、成功率区间和分布漂移。

## 生成预算与停止条件

```yaml
budget:
  episodes: 100000
  max_gpu_hours: 500
  max_storage_gb: 2000
  min_success_rate: 0.20
  max_corrupt_ratio: 0.001
```

达到存储、成本或质量阈值时，运行应可解释地终止，保留已完成 Episode 和终止原因。

## 数据生成后

```python
result = px.runs.wait(run)
dataset = px.datasets.get(result.dataset)

normalized = px.datasets.ingest(result.dataset, method="lerobot_v2_1")
snapshot = px.datasets.snapshot(normalized.outputs[0].artifact, method="dvc")
```

`datasets.ingest` 的 `quality_report` 通过后再创建快照和固定切分。不要让训练脚本
直接遍历运行临时目录。
