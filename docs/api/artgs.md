# ArtGS 视觉几何重建

`geometry.reconstruct:artgs` 使用仓库内原生 ArtGS，在当前 PhysReal-X 解释器中执行 CUDA 粗分解、运动类型预测、联合重建和验证视角渲染。源码、上游提交及修改见 `algorithms/artgs/SOURCE.toml`。

## 输入

文中的“验证视角”不参与训练或初始化点云，用于评估重建结果。它与训练视角来自同一次采集，不代表独立物理实测；对应字段为 `validation_*`，评估范围标记为 `held_out_views`。

ArtGS 与 CPU `rgbd_fusion` 共用 `MetricRGBDInput`，不需要另一套预打包目录 API：

```python
from physrealx import MetricRGBDInput

source = px.artifacts.put_json(
    MetricRGBDInput(
        camera_rig=metric_camera,
        enhanced_depth=enhanced_depth,
        reference_model=reference_model,
    )
)
operation = px.assets.reconstruct_geometry(
    source,
    method="artgs",
    config={
        "num_slots": 2,
        "state_streams": ["can.start.rgb", "can.end.rgb"],
        "reference_joint": "e_door_197.joint",
        "training_frames_per_state": 100,
        "max_initial_points": 1000000,
        "coarse_iterations": 5000,
        "joint_iterations": 3000,
        "reconstruction_iterations": 10000,
        "evaluation_resolution": 1,
        "seed": 2026,
    },
)
```

`num_slots` 必须显式指定为可观测运动组数。oven4 实拍展示箱体与旋转门两个运动组；参考 USD 中被固定的烤架与旋钮不会增加可学习运动组。

适配器消费公共 RGB-D 准备阶段已完成几何有效性判定的米制深度，并检查相机/RGB/深度身份、刚体变换、单位和数组大小。RGB、Mask、深度一起校正畸变；深度以最近邻转换为毫米 PNG。此输入清理不改变 ArtGS 的三阶段逻辑。每状态必须提供足够视角进行训练与独立视角划分，训练各阶段和初始化点云均排除验证视角。

自动关节类型预测和联合重建使用相同的 `point_loss_weight` 与 `metric_depth_loss_weight`。前者约束形变后的规范点云与对应状态的粗重建点云，后者仅覆盖前景掩码内的有效实测像素；背景零值不计入分母。类型预测文件同时记录优化得到的半程旋转角及 10° 判定阈值。

两运动组场景默认从第一阶段的两状态运动点云估计刚体变换，并以对称半程变换初始化第二阶段；若旋转拟合不能优于质心对齐平移，则保持平移初始化。薄片或近对称部件出现多个等价方向时，在几何误差带内选择最小旋转，避免 PCA 轴符号歧义造成近 180° 假运动。

提供 `reference_joint` 时，准备层先用训练视图的 Mask 与米制深度点云拟合配套 USD 表面，同时估计两个观测状态的关节位置；验证视图只执行距离门禁，不参与拟合。随后将状态间刚体运动变换回采集坐标并作为第二、三阶段的初值。RGB、Mask 和实测深度仍驱动训练损失，参考几何不作为逐像素监督或最终关节值。重建报告保存完整 `reference_depth_alignment`，状态为 `MODEL_CONDITIONED`；后续 `articulated_surface_icp` 直接绑定该结果，不重复拟合。当前实现仅支持一个根运动组和一个水平轴运动组。

第三阶段在无恢复检查点时沿用同一几何初始化，避免已经判定的旋转关节从近单位变换重新训练而丢失观测运动幅度。

`training_frames_per_state` 指定每状态的实际训练张数（整数 2–2048）；每三帧选一帧用于验证，
100 张训练图需要每状态 150 个有效输入视图，另 50 张只用于评估。适配器在原生训练前检查数量，
不足会报出各状态可用数，不自动降低目标。不指定此参数时按两状态较小的可用视图数均衡划分。

`max_initial_points` 默认 100 万，限制初始化高斯所用的深度点总数。点预算均衡分配给所有训练视图，验证视图的深度不参与初始化；这不减少训练图片数。两个状态分别从其训练 RGB-D 视图生成 `point_cloud_start.ply` 和 `point_cloud_end.ply` 并独立初始化，避免混合状态几何污染粗分解。报告中的 `initial_point_count` 记录两份种子点云的总点数。

## 输出

| 输出 | 内容 |
| --- | --- |
| `visual_geometry_bundle` | Gaussian 点云、形变权重、重载配置、运动组 Mesh、关节描述、五个连续外观状态，以及粗重建/类型预测/联合重建三阶段诊断图与封闭文件清单 |
| `geometry_reconstruction_report` | 训练集/验证集 PSNR、SSIM、LPIPS、前景 PSNR、轮廓 IoU，运动组数、坐标系、来源引用、训练配置与证据范围 |

五个状态由上游 `interpolate` 在未参与训练的实拍相机渲染，取值对应观测运动区间。未知关节限位、未观察部件运动与真实物性不能由外观指标推导。

`model/config.json` 保存网络与渲染配置，排除训练临时目录；配置统一为 JSON，删除旧的 `eval/Namespace` 文件路径。旋转轴使用带轴向约束的最小二乘求解，避免对必然奇异的 `I-R` 求逆；零运动不能发布可辨识关节轴。

接受自监督增强深度、USD 条件尺度及参考引导初始化，但报告保留 `MODEL_CONDITIONED`；独立物理验证字段始终不因图像指标而升级。无验证视角、参考配准未过门禁、空 Mesh、CUDA 错误或阶段超时均拒绝成功发布，不提供仅训练视角的兼容模式。

## 环境与来源

[安装说明](../getting-started/installation.md)列出统一依赖。原生扩展首次在当前环境编译，GPU/扩展不能执行时明确失败。算法主体为 Apache-2.0，整体运行依赖的 rasterizer/Simple-KNN 为非商业研究许可；方法声明 `LicenseRef-ArtGS-Research-Only`。

示例迭代和分辨率用于流程验收，不能作为论文最优效果或新的 SOTA 结论。

`joint_types` 可显式提供非根运动组的类型（如 `["hinge"]` 或 `["slide"]`），数量必须等于 `num_slots - 1`。提供时使用已知类型先验并省去自动类型分类阶段；不提供时从采集数据推断。轴、位置、幅度及几何仍需训练估计，报告记录 `joint_type_source` 与原始先验。当前 oven4 Notebook 不提供该参数，以完整执行三阶段训练。训练图像缓存在 CPU，逐帧传至 GPU，视图数量不变。
