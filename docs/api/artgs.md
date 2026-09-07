# ArtGS 视觉几何重建

`geometry.reconstruct:artgs` 使用仓库内原生 ArtGS，在当前 PhysReal-X 解释器中执行 CUDA 粗分解、运动类型预测、联合重建和留出视角渲染。源码、上游提交及修改见 `algorithms/artgs/SOURCE.toml`。

## 输入

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
        "state_streams": ["can.start.001.rgb", "can.end.001.rgb"],
        "coarse_iterations": 5000,
        "joint_iterations": 3000,
        "reconstruction_iterations": 10000,
        "evaluation_resolution": 1,
        "seed": 2026,
    },
)
```

`num_slots` 必须显式指定为可观测运动组数。cabinet2 实拍仅下层抽屉移动，柜体与静止上层抽屉合为一组；三个 USD 结构部件不能直接推导三个可学习运动组。

适配器检查相机/RGB/深度身份、刚体变换、米制单位、数组大小与有效性。RGB、Mask、深度一起校正畸变；深度以最近邻转换为毫米 PNG。每状态必须提供足够视角进行训练与独立视角划分。训练各阶段和初始化点云均排除留出视角。

## 输出

| 输出 | 内容 |
| --- | --- |
| `visual_geometry_bundle` | Gaussian 点云、形变权重、重载配置、运动组 Mesh、关节描述、五个连续外观状态及封闭文件清单 |
| `geometry_reconstruction_report` | 训练/留出 PSNR、SSIM、LPIPS、前景 PSNR、轮廓 IoU，运动组数、坐标系、来源引用、训练配置与证据范围 |

五个状态由上游 `interpolate` 在真实留出相机渲染，取值对应观测运动区间。未知关节限位、未观察部件运动与真实物性不能由外观指标推导。

`model/config.json` 保存网络与渲染配置，排除训练临时目录；配置统一为 JSON，删除旧的 `eval/Namespace` 文件路径。旋转轴使用带轴向约束的最小二乘求解，避免对必然奇异的 `I-R` 求逆；零运动不能发布可辨识关节轴。

接受自监督增强深度和 USD 条件尺度，但报告保留 `MODEL_CONDITIONED`；独立物理验证字段始终不因图像指标而升级。无留出视角、空 Mesh、CUDA 错误或阶段超时均拒绝成功发布，不提供仅训练视角的兼容模式。

## 环境与来源

[安装说明](../getting-started/installation.md)列出统一依赖。原生扩展首次在当前环境编译，GPU/扩展不能执行时明确失败。算法主体为 Apache-2.0，整体运行依赖的 rasterizer/Simple-KNN 为非商业研究许可；方法声明 `LicenseRef-ArtGS-Research-Only`。

示例迭代和分辨率用于流程验收，不能作为论文最优效果或新的 SOTA 结论。
