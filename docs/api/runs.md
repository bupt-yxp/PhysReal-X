# 运行与数据生成 API

`runs.submit(RunSpec) -> RunRecord` 同步执行已登记场景、任务、后端与策略。`runs.wait(run)` 读取终态 `RunResult`，`runs.get(run)` 读取运行记录，`runs.episodes(run)` 载入并验证不可变轨迹。

```python
from physrealx import MethodCapability, RunSpec

backend = px.methods.bind(
    MethodCapability.SIMULATION_RUN,
    "mujoco",
    config={
        "controlled_joints": ["cabinet/gui_drawer_1.joint"],
        "initial_positions": [0.0],
        "effort_limits": [40.0],
        "timestep_seconds": 0.002,
        "substeps": 25,
    },
)
policy = px.methods.bind(
    MethodCapability.POLICY_INFERENCE, "joint_target", config={"target": [0.238]}
)
run = px.runs.submit(
    RunSpec(
        scene=scene_ref,
        task=task_ref,
        backend=backend,
        policy=policy,
        episode_count=4,
        base_seed=2026,
        dataset_id="cabinet.simulation",
        dataset_version="1.0.0",
        record_signals=("joint.position", "joint.velocity", "contact.force"),
    )
)
dataset = px.datasets.get(px.runs.wait(run).dataset)
```

实际记录信号必须属于任务。后端配置固定在 MethodSpec 中；命令经有限增益、作用力限制后进入 MuJoCo，不直接改写物体状态。`passive_positions` 初始化未驱动关节，支持机器人与被动物体接触。

当前 MuJoCo 在一个状态上同步生成多相机 RGB/Depth/Mask/Event/点云/内外参及刚体位姿、外力/力矩，全部保留时间戳和语义标签。配置与信号表见[仿真指南](../user-guide/simulation-and-data-generation.md)。

## 生命周期

```text
解析精确资源 → 能力/单位预检 → reset → 策略动作 / 真实物理步进 → close
→ 核对 Episode 时间线/成败标签 → 原子发布 DatasetManifest
```

末帧没有待执行动作；运行失败不发布不完整数据集。无论正常、异常或中断，策略/session/渲染器均关闭。同一 run_id 和规格返回原结果，身份相同但规格不同会报冲突。

`RunSpec.stop_on_success` 默认 `True`，满足任务成功条件后结束该 episode。
采集世界模型的固定时长序列时可设为 `False`，继续到任务步数上限，并按末帧判断最终成功。
后端自身终止、任务显式终止条件和截断仍立即生效；此选项不会覆盖这些停止条件。
这一规则由运行编排层统一负责，MuJoCo 与 reference 后端保持一致。

核心 `ExecutionLifecycle` 还支持校验外部离线物理轨迹与渲染证据，但它不代替引擎执行。当前集成多模态通过 `runs.submit` 完成；独立 simulations/renders/events/pointclouds/episodes 方法入口保留为扩展协议，没有注册未实现的方法名。
