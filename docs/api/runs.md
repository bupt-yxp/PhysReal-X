# 运行与数据生成 API

`runs.submit(RunSpec) -> RunRecord` 同步执行已登记场景、任务、后端与策略。`runs.wait(run)` 读取终态 `RunResult`，`runs.get(run)` 读取运行记录，`runs.episodes(run)` 载入并验证不可变轨迹。

仿真引擎用 `backend` 选择，策略用 `policy` 指定。`method` 仍用于重建、配准等算法选择；
底层 `MethodSpec` 保存实现版本及配置，无需为每个后端新增一套数据类型。

快捷入口与 `runs.submit` 共用执行器、失败处理和数据集发布：

```python
run = px.simulations.run(
    scene_ref, task_ref,
    backend="isaac_sim",  # 或 "mujoco"
    config=control_config,
    policy=policy,
    dataset_id="simulation.dataset",
    episode_count=4,
    base_seed=2026,
)
```

以下为需要固定 `run_id`、标签等完整配置时的调用：

```python
from physrealx import MethodCapability, RunSpec

backend = px.methods.bind(
    MethodCapability.SIMULATION_RUN,
    "mujoco",
    config={
        "controlled_joints": ["pusher/extension"],
        "initial_positions": [0.0],
        "effort_limits": [40.0],
        "timestep_seconds": 0.002,
        "substeps": 25,
    },
)
policy = px.methods.bind(
    MethodCapability.POLICY_INFERENCE, "joint_target", config={"target": [0.195]}
)
run = px.runs.submit(
    RunSpec(
        scene=scene_ref,
        task=task_ref,
        backend=backend,
        policy=policy,
        episode_count=4,
        base_seed=2026,
        dataset_id="oven4.simulation",
        dataset_version="1.0.0",
        record_signals=("joint.position", "joint.velocity", "contact.force"),
    )
)
dataset = px.datasets.get(px.runs.wait(run).dataset)
```

实际记录信号必须属于任务。后端配置固定在 MethodSpec 中；命令经有限增益、作用力限制后进入物理引擎，不直接改写物体状态。`passive_positions` 初始化未驱动关节，支持机器人与被动物体接触。

MuJoCo 与 Isaac Sim 在同一物理状态上同步生成多相机 RGB/Depth/Mask/Event/点云/内外参及刚体位姿、外力/力矩，全部保留时间戳和语义标签。配置与信号表见[仿真指南](../user-guide/simulation-and-data-generation.md)。

## 生命周期

```text
解析精确资源 → 能力/单位预检 → reset → 策略动作 / 真实物理步进 → close
→ 核对 Episode 时间线/成败标签 → 原子发布 DatasetManifest
```

末帧没有待执行动作；运行失败不发布不完整数据集。无论正常、异常或中断，策略/session/渲染器均关闭。同一 run_id 和规格返回原结果，身份相同但规格不同会报冲突。

`RunSpec.stop_on_success` 默认 `True`，满足任务成功条件后结束该 episode。
采集世界模型的固定时长序列时可设为 `False`，继续到任务步数上限，并按末帧判断最终成功。
后端自身终止、任务显式终止条件和截断仍立即生效；此选项不会覆盖这些停止条件。
这一规则由运行编排层统一负责，MuJoCo、Isaac Sim 与 reference 后端保持一致。

核心 `ExecutionLifecycle` 还支持校验外部离线物理轨迹与渲染证据，但它不代替引擎执行。`simulations.run(backend=...)` 已接通统一执行器；旧的 `simulations.run(method=...)` 占位接口和对应虚构输出契约已删除。独立 renders/events/pointclouds/episodes 方法入口仍为扩展协议。
