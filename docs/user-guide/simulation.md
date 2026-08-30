# 仿真运行与后端

PhysReal-X 使用统一后端协议执行仿真，但不会强行抹平后端能力差异。运行提交前，平台根据任务要求和工作节点能力进行匹配。
核心仓库只内置确定性 Reference backend 用于契约回归，不内置 MuJoCo、Isaac Sim、Genesis、Gazebo 或真机后端；真实物理结果必须来自独立安装、通过探针且留存运行证据的适配器。
后端名称只是方法选择：对应独立 worker 必须已安装、固定版本并通过探针；
核心仓库不会用硬编码轨迹代替真实后端。

## 如何选择后端

| 需求 | 推荐后端方向 |
| --- | --- |
| USD、RTX、多相机、高保真铰接操作 | Isaac Sim / Isaac Lab |
| 大规模向量化强化学习 | Isaac Lab 或高吞吐后端 |
| 可变形体、材料或多物理研究 | Genesis |
| 轻量控制、经典动力学和快速 CI | MuJoCo |
| 领域契约与测试 | Reference backend |
| 真实机器人部署与回放 | ROS 2 / robot backend |

选择依据必须来自能力声明和基准结果，不应在业务代码里判断后端名称。

## 能力声明

工作节点注册时报告：

```python
BackendCapabilities(
    rigid_body=True,
    articulation=True,
    deformable=False,
    contact_force=True,
    batched=True,
    deterministic_reset=True,
    observation_keys=("rgb", "depth", "joint_state", "contact_force"),
    action_kinds=("joint_position", "ee_delta_pose"),
    devices=("cuda:0",),
)
```

若任务需要后端无法提供的力觉或动作类型，平台在创建场景前快速失败。

## 运行规格

```python
from physrealx import MethodCapability, RunSpec

backend = px.methods.bind(
    MethodCapability.SIMULATION_RUN,
    "<installed-simulation-method>",
    config={"physics_dt": 0.002, "control_dt": 0.02},
)
policy = px.methods.bind(
    MethodCapability.POLICY_INFERENCE,
    "motion_planner",
)

run = px.runs.submit(
    RunSpec(
        scene=scene,
        task=task,
        backend=backend,
        policy=policy,
        episode_count=10_000,
        base_seed=2026,
        dataset_id="drawer.rollouts",
        dataset_version="1.0.0",
        record_signals=("camera.front.rgb", "robot.state", "contact.force"),
    )
)
```

运行提交后，规格中的可读引用会解析为精确资源摘要、仿真后端版本与策略版本，形成不可变执行计划。

## 会话生命周期

```text
匹配工作节点
  → 拉取并验证资产制品
  → materialize 场景
  → 创建后端会话和策略实例
  → reset(seed + episode_index)
  → observe → act → step → record
  → Episode 终止与校验
  → 重置下一 Episode
  → 关闭策略、会话和临时资源
```

昂贵场景和模型在同一运行中复用；每条 Episode 只重置必要状态。会话关闭和临时资源清理必须幂等。

## 时间步

平台区分：

- `physics_dt`：物理求解步长；
- `control_dt`：策略与控制器更新周期；
- `render_dt`：渲染采样周期；
- `sensor_dt`：每个传感器自身采样周期。

不要假设四者相等。Episode 保存控制时间线和各传感器采样时间，数据导出器再根据训练格式执行对齐、插值或窗口采样。

## 确定性与复现

固定种子至少控制：

- 场景和任务采样；
- 资产实例和初始状态；
- 传感噪声与域随机化；
- 策略探索噪声；
- 后端可控随机源。

GPU 并行求解和浮点运算可能带来位级差异，因此平台区分：

- **规格可复现**：输入、版本、种子和环境一致；
- **统计可复现**：指标在声明容差内一致；
- **位级确定**：摘要完全一致，仅在后端明确保证时声明。

## 并行执行

`parallelism` 表示期望并行度，不等于强制使用同一 GPU。调度器依据：

- 后端是否支持 batched/vectorized execution；
- 每环境显存与 CPU 内存；
- 传感器和渲染开销；
- 数据写入带宽；
- 工作节点标签和资源配额。

Episode ID 与种子不依赖实际调度顺序，扩缩容不会改变逻辑身份。

## 取消、重试与恢复

- 取消：停止分配新 Episode，正在写入的 Episode 完成原子提交或丢弃；
- 重试：只重试可恢复失败，保持 Episode 逻辑 ID 和尝试次数；
- 恢复：从已发布 Episode 集继续，不重复发布相同索引；
- 节点失联：租约到期后回收任务，孤立制品不进入数据集清单。

数据集只有在满足规格中的 Episode 数、质量门和预算策略后发布。

## 监控

```python
record = px.runs.get(run)
print(record.status, record.started_at, record.completed_at)
```

当前本地执行器是同步的，没有 `watch` 流式接口。外部调度扩展可以上报环境步速、渲染帧率、
Episode 成功率、GPU 利用率、队列延迟和制品写入吞吐，但不得伪造核心客户端方法。
