# 功能验收与证据范围

本页区分当前源码与历史执行证据。旧对象与旧方法回执保存在 `.physrealx/example-history/`，不作为当前 laptop1 Notebook 的通过证明。

## 当前入口

`examples/real_assets_and_digital_twin/demo.ipynb` 使用 laptop1 的首末 RGB-D-Mask 状态与配套逆向建模 USD。参考模型含机身、屏幕和一个旋转关节。当前流程为：

```text
RGB/Depth/Mask 观测导入 → 相机恢复 → USD 条件米制对齐 → 深度增强
→ 训练帧深度到 USD 表面配准 → ArtGS 重建 → 配准绑定
→ 固定 DTArt 指标 → 对比动画与导出
```

Notebook 使用全部已注册 RGB-D-Mask 视图；重建固定选择每状态 100 张训练图和 50 张验证图。帧选择、Mask 校验与绑定、几何计算、质量门及导出都由项目 API 完成，Notebook 只声明输入、方法和业务参数。

## 当前执行证据

2026-09-16 曾在现有 PhysReal-X Conda 环境与 RTX 4090 上，按依赖顺序用干净内核完整执行六个 Notebook。下表摘要与当时源码一致；后续 Notebook 与方法实现已经更新，因此该批回执只保留为历史证据，不证明当前六模块版本已全量重跑。实际执行总计 11,130.37 秒：

| 模块 | 代码单元 | 用时（秒） | 当时源码 SHA-256 |
| --- | ---: | ---: | --- |
| 真实资产与数字孪生 | 9 | 1,914.71 | `3159dbd4daa93aa7743b8a5bf9e4ea647cf8a6ed817ad0dbf10eb64ad44d1261` |
| 物理资产 | 7 | 1,631.77 | `62c8ac6b496570c4ccfa6946c33414ad4d3e21ac377d3fbc516749566e9b9bd3` |
| 场景与任务 | 8 | 117.22 | `02ddeb3ba216b003bbdaddd3e0b6155817705829695592e9ddf13749c1c4cc8e` |
| 仿真与数据 | 9 | 2,794.17 | `099ab8a93d8285ad1c7305cb5c48be9f52960fcaa032a65e531c5005450e90ae` |
| 数据与训练 | 11 | 1,582.71 | `7efe304873e575a3585e4b1f1fdcd2d3e3f452f068d732a151eaa283e38c975b` |
| 评测与反馈 | 10 | 3,089.78 | `dda675c5b94c39873931763ef9e25a94d660f3ed963f369501744a9a74f6a2f8` |

每个模块的 `output/demo.executed.ipynb`、`output/execution.json` 与 `output/results.json` 分别保存执行副本、逐单元回执和业务结果。续跑执行器会先核验上游三类证据及源码哈希；不能用旧输出拼接当前成功状态。

2026-09-17 已单独用干净内核执行当前场景与任务 Notebook：平台按 `home` 场景类型从 PartNet-Mobility 资产目录选择并校验物体，导入 MuJoCo Menagerie 官方 Franka Emika Panda，使用 7 关节逆运动学计划，在 MuJoCo 中完成接近、接触和关闭笔记本屏幕任务，并生成 16 帧 Blender 动画及场景、任务、轨迹、成功/失败样本四类索引。任务成功，最大关节跟踪误差为 0.1897 rad，峰值接触力为 135.8 N。

同日已执行当前物理仿真 Notebook：使用模块三发布的 Panda 七关节任务实例和轨迹，生成 2 条成功专家 Episode 和 2 条无动作失败 Episode，每条 201 帧；两台相机及 27 个信号共产生 12,864 个同步数组，数据质量检查通过。同步 GIF 以记录的刚体位姿驱动单一 Blender PBR 场景；overview 与 front 共享资产、材质和光照，front 同步输出 RGB、米制 Depth、实例/Link Mask 和相机位姿，XYZRGB 点云由同一组 RGB-D 反投影。模块五至六尚未针对该新版数据重跑。

历史 oven4 执行中，ArtGS 2.5.0 使用每状态 100 张训练图和 50 张验证图，输出静态根与一个旋转部件。验证 PSNR 27.138 dB、前景 PSNR 25.242 dB、SSIM 0.9381、LPIPS 0.0597、轮廓 IoU 0.9520。训练前参考配准的最大验证中位距离为 4.31 mm、P90 为 8.19 mm；拟合状态为 start 90.00°、end 17.48°，start 状态触及 USD 上限并以 `joint_limit_saturated=1` 保留。

按统一 DTArt 协议评估：Axis Ang 0.715°、Axis Pos 0.1172 × 10⁻¹ m（11.72 mm）、Part Motion 9.385°；起始/结束状态的运动部件 CD 为 0.713/0.617、静态 CD 均为 0.301、整体 CD 为 0.331/0.296，CD 单位均为 10⁻³ m²。这些数值属于 USD 参考引导的模型条件评估，不是独立实物真值。

反馈模块从 5 条失败轨迹生成同初态恢复示范并实际再训练 ACT；相同 5 个新种子的成功率由 0.4 提升至 1.0。

## 软件验证

当前改动已通过：

- Ruff 全量源码与测试检查；
- Mypy 75 个源码文件；
- `MUJOCO_GL=egl` 下 366 项测试通过，3 项条件测试跳过，包含 8 项 Notebook 执行器测试；
- MkDocs 严格构建；
- wheel 无隔离构建及内嵌源码检查；
- `pip check` 依赖一致性；
- 数据集 Mask 数量、命名、分辨率、二值范围及非空性检查。

## PPT 功能对照

需求依据为工作区《具身方向.pptx》和《具身方向v3.pptx》。团队介绍、论文演示数字与未开源实现不作为软件能力。

| 服务 | 当前实现 | 当前证据边界 |
| --- | --- | --- |
| 真实资产与数字孪生 | ArtModeling RGB/Depth/Mask、COLMAP、米制对齐、参考引导 ArtGS/DTArt、USD 配准、关节/CD 评估 | 当前入口使用 laptop1；最近一次六模块全量回执仍属于历史 oven4 版本 |
| 物理资产 | CoACD、显式物性先验、USDZ、MuJoCo 回读 | 质量、摩擦、刚度、阻尼没有独立实测时只能标记为先验 |
| 场景与任务 | PartNet-Mobility 房间布局、官方 Franka Panda、多关节逆运动学、MuJoCo 接触执行与 Blender 动画 | 当前验证明确的单关节物体任务，不代表任意语言规划 |
| 仿真与多模态 | MuJoCo RGB/Depth/Mask/Event/Pose/点云/力觉 | 是真实仿真后端；传感数据仍为仿真数据 |
| 数据与训练 | 成功/失败分层快照、ACT、DINO-WM、数据/策略/预测可视化 | 历史 oven4 版本已实跑；当前 laptop1 源码仍需重新执行，不证明未见对象泛化 |
| 评测与反馈 | 轨迹指标、失败归因、数据回流、ACT 再训练及同种子配对评测 | 初始 0.4、更新 1.0；没有真机设备，因此没有现场闭环证据 |

## 证据边界

- 逆向建模 USD 提供尺度、几何、关节与材质参考，不是独立物性实测。
- 数据集 Mask 属于版本化输入；USD 不参与生成目标轮廓。
- `status=COMPLETED/PASSED` 表示软件操作和声明的质量门完成，不表示论文指标达标。
- 当前没有真机设备、独立质量/受力记录或现场控制日志；真机闭环不属于本轮离线验收。
- `assets.evaluate_reconstruction` 对所有重建方法固定使用 DTArt 的 Axis Ang、Axis Pos、Part Motion 与平方 CD 定义；指标不属于 DTArt 重建方法的专属输出。

完整验收要求源码摘要、输入制品、方法描述符、权重摘要、输出制品和 Notebook 回执相互一致。修改 Notebook 或方法实现后，旧回执自动降为历史证据。
