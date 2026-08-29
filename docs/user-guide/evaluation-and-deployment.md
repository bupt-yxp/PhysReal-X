# 评测、真机部署与失败反馈

完整闭环采用逐级门禁：

```text
离线回放 → 冻结仿真评测 → shadow → 限速/限力 canary → 真机运行
→ MCAP 留证 → 故障确认 → 定向数据集 → 再训练 → 全量回归
```

## 冻结评测

```python
report_ref = px.evaluations.submit(
    resource_id="drawer.policy.heldout",
    version="1.0.0",
    model=model_ref,
    run=heldout_run,
    episode_count=200,
    base_seed=100_000,
)
report = px.evaluations.get_report(report_ref)
```

训练与评测不得共用随机种子或资产/场景切分。总体成功率之外，还需报告逐任务置信区间、完成时间、
接触安全、推理延迟和未见资产最差分组。

## 真机门禁

1. 模型输入输出与机器人 Schema、频率、单位、frame 和关节顺序完全一致；
2. 急停、watchdog、工作空间、速度、力矩、力和碰撞限制独立于学习策略；
3. 先做日志回放和 shadow，再做有人看护的限速/限力 canary；
4. rosbag2 + MCAP 保存统一时钟下的观测、动作、TF、诊断和任务判据；
5. 任一关键安全事件阻止扩大部署范围。

## 故障归因

固定原因族覆盖传感/同步/标定、感知/姿态/语义、策略无进展/振荡、控制跟踪/超时、接触滑移/
卡死/超力/碰撞、任务判据及基础设施。每条结论必须指向发生时刻和证据；模型 critic 的建议需经
规则或人工确认，无法确认时保留 `unknown`。

## 定向再训练

从确认失败和 OOD 切片建立新的不可变数据集版本，保留父 Episode、模型、资产、场景、任务和方法
来源。再训练只有在独立 failure holdout 改善且旧回归集不退化时才可晋级。
