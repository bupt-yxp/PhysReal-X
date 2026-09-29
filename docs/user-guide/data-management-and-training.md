# 数据管理与模型训练

模块五执行统一 Episode 数据集、分组快照、ACT/IQL 训练、独立仿真评测及 DINO-WM 多步/反事实推演，并准备 OpenVLA-OFT 与 TD-MPC2 的官方运行接口。

<!-- demo-preview:data_management_and_training -->

<!-- demo-image:data_management_and_training:dataset-preview.png -->

## 数据与快照

`datasets.ingest(..., method="episode_dataset")` 合并相同观测/动作 Schema 的不可变数据集，核查实际 NPY 的 dtype/shape、时间戳、源标签和 Episode 摘要。`datasets.snapshot(..., method="episode_split")` 发布核心 `DatasetSnapshot`，包含精确数据、代码源树、环境及 `split.train/validation/test`。`stratify_success=True` 在保持场景、任务和种子分组隔离的同时保留成功/失败分层；训练分组不足以覆盖已有结果类型时明确失败。

真实数据按 capture_id 分组；仿真按场景版本、任务版本与种子分组。所有帧与窗口留在各自 Episode 内。该划分验证新 episode/种子，不代表未见资产泛化；当前示例仅覆盖一个笔记本关节任务。

Episode 保留任务语言、观测、动作、下一状态、奖励、终止与成败信息。ACT 读取成功专家 Episode；IQL 读取训练分区内全部成功/失败 Episode。验证与测试分区均不进入训练。

## ACT

`policy.train:act` 与 `policy.inference:act` 使用 LeRobot v0.3.3 的官方 CVAE/Transformer ACT 网络，来源固定在 `algorithms/act/SOURCE.json`。`datasets.training_subset` 统一选取训练分组中的成功专家 episode，并检查补充数据的分组隔离；训练侧计算归一化，保存 safetensors 和完整配置。

默认推理执行完整训练动作块，再根据新观测重新规划；不会每步仅取动作块第一项而反复停在慢启动阶段。模型重载后由 `evaluations.submit` 在独立种子执行闭环。示例小型 ResNet18/Transformer 配置不表示论文性能复现。

## IQL 与官方训练接口

`policy.train:iql` 使用固定版本的官方 IQL 算法定义，在平台内以 PyTorch 连续动作适配实现 expectile value、双 Q、目标网络和优势加权策略训练。输出 safetensors `ModelManifest`，并经同一 `evaluations.submit` 闭环评测。

`training.prepare_interface(snapshot, method=...)` 为 OpenVLA-OFT 和 TD-MPC2 检查所需信号，固定官方源码提交，记录字段映射、数据适配契约、资源要求、启动参数和预期产物。接口状态为 `PREPARED`，表示交接契约已验证；只有外部官方运行完成并回传检查点与评测报告后才能标记训练完成。

## DINO-WM

`world_models.train(snapshot, encoder, method="dino_wm")` 使用冻结的官方 DINOv2 ViT-S/14 编码器与 DINO-WM 官方因果 ViT。DINO-WM ViT 预测视觉特征；独立状态残差网络依据完整本体状态和动作预测机械状态，避免视觉误差累积后反过来推动关节。RGB 解码保留上下文图像细节并预测变化量。这些平台适配明确写入模型配置及代码来源。

返回核心 `WorldModelBundle` 与 `TrainingReport`，与核心训练生命周期共用快照/代码/环境/权重身份。编码器权重首次从 Meta 官方固定地址下载、校验架构并转为 safetensors；此后复用当前工作区。

`world_models.rollout(model, context, actions, method="dino_wm")` 仅读取起点及之前的上下文，核对历史动作、单位和频率，生成未来 RGB、状态与时间戳。模块五使用 history=1、100 步可微滚动训练。状态头不依赖视觉分支，两者分别按独立 validation 组选择检查点；视觉分支同时校准有界残差增益，抑制长时递归漂移。最终组合的权重、选择步数、增益和曲线写入训练报告。test 组只用于多个起点的 100 步评估；正常动作与零动作分别对照完整的五秒 MuJoCo 真值，并与保持初始状态、特征和图像的基线比较。测试窗口不参与检查点或增益选择。

超出记录轨迹的未来没有真值，不计入精度。状态预测必须优于 persistence 基线；潜变量和 RGB 指标用于诊断，校准选择零视觉残差时会与 persistence 持平。潜在特征变化不能当作概率置信区间或物理安全保证。混合单位状态误差报告归一化量，位置误差另以米报告。

## 可视化证据

模块五由项目服务生成四类可视化制品：

- `datasets.preview_snapshot`：RGB、深度、Mask、事件、同步序列和数据划分；
- `training.preview_policy`：ACT/IQL 损失曲线与独立闭环结果；
- `training.preview_interfaces`：OpenVLA-OFT/TD-MPC2 的信号映射、资源要求和执行状态；
- `world_models.preview_rollout`：留出 Episode 的 MuJoCo RGB/状态真值与 DINO-WM 预测动画。

Notebook 只展示和导出制品，不重复数据选择、指标计算或绘图逻辑。

下一步见[评测与反馈](evaluation-and-feedback.md)。
