# 评测与反馈闭环

模块六从模块五的 ACT 权重开始，在独立仿真种子上评测，并对保留的真实 MuJoCo 失败轨迹执行遥测归档、诊断、样本回流和同初态再生成。

## 评测和日志

`evaluations.submit` 固定模型、场景、任务、后端和种子，使用重载权重实际控制引擎。`telemetry.record(..., method="episode_telemetry")` 消费指向数据集的 `RunEvidence`，重新读取原始 Episode 与 TaskSpec，计算目标误差、轨迹跟踪 RMSE、运动范围、作用力及采样接触峰值。

归档接受已规范化为核心 Episode 的实机数据，但保留 `source=real` 与 capture_id。仿真记录保留 `source=simulator`；归档行为本身不是现场部署。外部机器人原始日志必须先提供真实 Schema、时钟、动作语义与相应导入适配器。

## 失败回流

`failures.analyze(..., method="trajectory_diagnostics")` 检查记录中的接触异常、命令目标不符、动力学跟踪异常与进度不足，并给出相应检查或采集建议。它重新计算不可变轨迹指标，拒绝篡改后的遥测。

这些分类属于可观察症状；因果判断需要受控干预或独立测量。`feedback.build_dataset(..., method="failure_episodes")` 保留失败 Episode 的 ID、标签、来源和原始制品，同时输出再生成种子及关联关系。

模块六使用失败时相同的场景、任务、后端参数与种子，执行专家轨迹生成新的成功样本。原失败集不覆盖、不改标签。`datasets.training_subset(snapshot, augmentations=...)` 只接受成功专家示范，并拒绝任何与原 validation/test 场景分组重叠的补充数据。

模块六确实调用 `training.submit` 产生新的 ACT 权重；恢复数据和动作窗口调整共同记录，不能将改善单独归因于新增数据。最后使用未参加反馈的另一组相同种子配对比较新旧模型，验收物体目标、成功率、接触时间及峰值。

## 现场机器人边界

核心 `robot_agent.py` 保留机器人接口、安全约束、影子/受限执行和看门狗协议；当前仓库没有实际 ROS 2 硬件适配器，也未提供现场机器人、控制器配置或部署日志。因此 Notebook 不调用不存在的部署方法，亦不将推杆仿真和协议测试标为机械臂真机验收。

独立质量、摩擦/力觉测量、机器人规格与原始日志到位后，可沿已有物理校准和反馈契约接入；软件与仿真完成状态保持独立。见[功能验收](../acceptance.md)。
