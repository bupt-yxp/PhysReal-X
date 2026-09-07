# 数据管理与模型训练

模块五实际执行统一 Episode 数据集、分组快照、ACT 训练重载、独立仿真评测及 DINO-WM 多步/反事实推演。所有步骤使用现有 PhysReal-X 环境。

## 数据与快照

`datasets.ingest(..., method="episode_dataset")` 合并相同观测/动作 Schema 的不可变数据集，核查实际 NPY 的 dtype/shape、时间戳、源标签和 Episode 摘要。`datasets.snapshot(..., method="episode_split")` 发布核心 `DatasetSnapshot`，包含精确数据、代码源树、环境及 `split.train/validation/test`。

真实数据按 capture_id 分组；仿真按场景版本、任务版本与种子分组。所有帧与窗口留在各自 Episode 内。该划分验证新 episode/种子，不代表未见资产泛化；cabinet2 只有一个对象。

Episode 保留任务语言引用、观测、动作、下一状态、奖励、终止与成败信息，供模仿学习、RL、VLA 和世界模型读取。当前执行的是 ACT 与 DINO-WM；没有把 RL/VLA 数据接口写成已训练的 RL/VLA 模型。

## ACT

`policy.train:act` 与 `policy.inference:act` 使用 LeRobot v0.3.3 的官方 CVAE/Transformer ACT 网络，来源固定在 `algorithms/act/SOURCE.json`。`datasets.training_subset` 统一选取训练分组中的成功专家 episode，并检查补充数据的分组隔离；训练侧计算归一化，保存 safetensors 和完整配置。

默认推理执行完整训练动作块，再根据新观测重新规划；不会每步仅取动作块第一项而反复停在慢启动阶段。模型重载后由 `evaluations.submit` 在独立种子执行闭环。示例小型 ResNet18/Transformer 配置不表示论文性能复现。

## DINO-WM

`world_models.train(snapshot, encoder, method="dino_wm")` 使用冻结的官方 DINOv2 ViT-S/14 编码器与 DINO-WM 官方因果 ViT。DINO-WM ViT 预测视觉特征；独立状态残差网络依据完整本体状态和动作预测机械状态，避免视觉误差累积后反过来推动关节。RGB 解码保留上下文图像细节并预测变化量。这些平台适配明确写入模型配置及代码来源。

返回核心 `WorldModelBundle` 与 `TrainingReport`，与核心训练生命周期共用快照/代码/环境/权重身份。编码器权重首次从 Meta 官方固定地址下载、校验架构并转为 safetensors；此后复用当前工作区。

`world_models.rollout(model, context, actions, method="dino_wm")` 仅读取起点及之前的上下文，核对历史动作、单位和频率，生成未来 RGB、状态与时间戳。模块五使用 history=1、16 步可微滚动训练；按独立 validation 组选择检查点，再在 test 组的多个起点进行 100 步评估。正常动作与零动作分别对照完整的五秒 MuJoCo 真值，并与保持初始状态、特征和图像的基线比较。测试窗口不参与选模。

超出记录轨迹的未来没有真值，不计入精度。潜在特征变化只是未校准诊断量，不能当作概率置信区间或物理安全保证。混合单位状态误差报告归一化量，位置误差另以米报告。

下一步见[评测与反馈](evaluation-and-feedback.md)。
