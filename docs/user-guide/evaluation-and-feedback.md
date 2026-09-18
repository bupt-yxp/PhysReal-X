# 评测与反馈闭环

模块六从模块五的策略权重开始，归档部署 Episode，诊断失败、生成恢复示范、重新训练，并使用独立种子比较模型更新效果。

## 评测和日志

`evaluations.submit` 固定模型、场景、任务、后端和种子。`telemetry.record(..., method="episode_telemetry")` 消费 `DeploymentEvidence`，重新读取不可变 Episode 与 TaskSpec，计算目标误差、轨迹跟踪 RMSE、接触稳定性和感知数据质量。

真机归档必须同时满足 `source=real`、独立 `capture_id` 和可验证的 `physrealx.deployment-report/1`；缺少部署计划、原始遥测或安全事件时拒绝真机标记。仿真归档保留 `source=simulator`，且不得附加真机部署报告。

## 失败回流

`failures.analyze(..., method="trajectory_diagnostics")` 检查感知有效性、接触异常、规划目标不符、控制跟踪异常与超时。它重新计算不可变轨迹指标，拒绝篡改后的遥测。

症状分类不直接声明原因。`feedback.analyze_policy_intervention` 对相同来源、场景、任务、后端和种子的失败/专家恢复轨迹做配对，只改变策略后成功且满足任务谓词时，才确认策略因素。`feedback.build_dataset(..., method="failure_episodes")` 保留失败 Episode 的原始标签与来源。

模块六使用失败时相同的场景、任务、后端参数与种子，执行专家轨迹生成新的成功样本。原失败集不覆盖、不改标签。`datasets.training_subset(snapshot, augmentations=...)` 只接受成功专家示范，并拒绝任何与原 validation/test 场景分组重叠的补充数据。

模块六通过训练器检查点继续训练 ACT，保持原网络结构和预测时域，并把父权重写入模型血缘。`feedback.assess_policy_update` 在相同场景、任务、后端、来源和种子上配对比较新旧策略：成功率下降返回 `REJECT`；成功率未提高，或目标误差、跟踪误差、接触稳定性越过护栏时返回 `COLLECT_MORE_DATA`；成功率提高且护栏均通过才返回 `PROMOTE`。评测面板展示这些指标、失败类别、校准状态和门控结果；同步 GIF 展示失败策略、专家恢复和更新策略。

## 现场机器人边界

核心 `robot_agent.py` 提供机器人接口、安全约束、影子/受限执行和看门狗协议。Notebook 当前输入为 MuJoCo Episode，因此校准计划返回 `AWAITING_REAL_EVIDENCE`，不会把仿真轨迹当作真机数据。

真实部署日志到位后，`feedback.prepare_physics_calibration` 从两个独立采集生成系统辨识证据，再由 `assets.calibrate_physics(..., method="bounded_inverse_dynamics")` 估计惯量、黏性阻尼和库仑摩擦，并使用独立 holdout 验证。见[功能验收](../acceptance.md)。
