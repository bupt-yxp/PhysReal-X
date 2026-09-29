# 评测与反馈闭环

模块六从模块五的策略权重开始，归档部署 Episode，诊断失败、生成恢复示范、重新训练，并使用独立种子比较模型更新效果。

<!-- demo-preview:evaluation_and_feedback -->

<!-- demo-image:evaluation_and_feedback:feedback-dashboard.png -->

## 评测和日志

`evaluations.submit` 固定模型、场景、任务、后端和种子。`telemetry.record(..., method="episode_telemetry")` 消费 `DeploymentEvidence`，重新读取不可变 Episode 与 TaskSpec，计算目标误差、轨迹跟踪 RMSE、接触稳定性和感知数据质量。

策略评测统一标记为 `policy_evaluation`，不继承上游的 `expert_demonstration` 标签。专家训练选择也会拒绝带模型推理标识的 Episode，防止策略输出被误当成专家答案。

真机归档必须同时满足 `source=real`、独立 `capture_id` 和可验证的 `physrealx.deployment-report/1`；部署报告中的 `telemetry` 引用 `TelemetryManifest`；Episode 的 `deployment_report` 与 `raw_telemetry` 分别关联该报告及其原始 MCAP，并校验记录时间范围；缺少部署计划、原始遥测或安全事件时拒绝真机标记。仿真归档保留 `source=simulator`，且不得附加真机部署报告。

## 失败回流

`failures.analyze(..., method="trajectory_diagnostics")` 检查感知有效性、接触异常、规划目标不符、控制跟踪异常与超时。它重新计算不可变轨迹指标，拒绝篡改后的遥测。

症状分类不直接声明原因。`feedback.analyze_policy_intervention` 对相同来源、场景、任务、后端和种子的失败/专家恢复轨迹做配对，在仿真中确认初始关节状态一致，只改变策略后成功且满足任务谓词时，才确认策略因素；真机重跑只记录关联，不直接声明因果。`feedback.build_dataset(..., method="failure_episodes")` 保留失败 Episode 的原始标签与来源。

模块六先在失败时相同的条件下执行专家恢复，再通过 MuJoCo 初始关节姿态扰动补充示范。当前专家是 `joint_sequence` 轨迹控制器，变化范围限定在该轨迹可执行的条件附近；改变物体位置或任务目标时需重新规划专家轨迹。采集结果保留所有成功和失败 Episode，仅成功示范进入训练。`feedback.preview_demonstrations` 展示采集结果、实际初始姿态分布与关节运动。

`datasets.training_subset(snapshot, augmentations=..., excluded_datasets=...)` 拒绝与原 validation/test 或额外保留评测集重叠的场景分组；模型验收还会追溯全部父模型的训练数据。独立专家验收轨迹只用于质量比较，不进入训练。如果将旧评测失败条件用于训练，必须重新保留评测条件。

ACT 微调使用完整父模型清单作为训练器检查点，复用权重、信号顺序、网络结构及归一化参数；裸权重不作为微调输入。重复提交仍校验数据选择与模型权重。

`feedback.assess_policy_update(initial, updated, reference_dataset=...)` 在相同场景、任务、后端、来源、种子及初始关节状态上比较新旧策略。执行器目标误差、跟踪误差和接触波动只比较同条件下双方均成功的执行：提供专家验收集时与专家比较，否则与成功的原策略比较，不以静止失败轨迹作为接触质量基准。没有可比样本时指标为 `null`，不伪造零误差。

默认门控要求至少 20 次独立评测、候选成功率达到 80%、至少 3 对成功执行质量对照，并满足误差及接触护栏；阈值可按实验设置调整，不代表统计置信度或物理安全认证。成功率下降返回 `REJECT`；证据不足、成功率未提高或护栏未满足返回 `COLLECT_MORE_DATA`；全部通过才返回 `PROMOTE`。Demo 默认采集 6 条补充示范、保留 3 个评测条件，用于运行展示，因此不会因少量成功直接晋升模型。

评测面板展示成功次数、验收数量要求、执行器误差、接触指标及门控状态；同步 GIF 展示失败策略、专家恢复和更新策略，使用共同时间与纵轴范围，提前结束的轨迹保持末帧。全部采集、专家验收集、训练集、模型和图像均由 `output/results.json` 索引。

## 现场机器人边界

核心 `robot_agent.py` 提供机器人接口、安全约束、影子/受限执行和看门狗协议。Notebook 当前输入为 MuJoCo Episode，因此校准计划返回 `AWAITING_REAL_EVIDENCE`，不会把仿真轨迹当作真机数据。

真实部署日志到位后，`feedback.prepare_physics_calibration` 从两个独立采集生成系统辨识证据（每条至少 16 个样本）。Episode 必须声明 `gravity_compensated=true` 与 `isolated_joint`，且关节索引、时间和信号单位通过校验，再由 `assets.calibrate_physics(..., method="bounded_inverse_dynamics")` 估计惯量、黏性阻尼和库仑摩擦，并使用独立 holdout 验证。
