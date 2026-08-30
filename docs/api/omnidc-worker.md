# OMNI-DC 深度增强 worker

该 worker 只封装 [OMNI-DC 官方实现](https://github.com/princeton-vl/OMNI-DC)，不复制模型代码，
也不把插值或形态学补洞伪装为网络推理。工作区中只有官方源码、权重、探针和
部署证明全部通过时，`omni_dc` 才会标记为就绪。

## 输入契约

`CalibratedRGBDBundle/1` 每帧必须包含：

- RGB、米制稀疏深度和不参与网络输入的互斥留出深度；
- 完整相机内外参、统一坐标系和已解析米制尺度；
- 原始采集清单、标定证据及全部依赖的 SHA-256；
- 与操作 `MethodSpec.checkpoint` 完全一致的 checkpoint manifest。

当前适配器只接受单权重闭包的官方 v1.0。v1.1 还依赖 Depth-Anything-V2
权重，在该附加权重进入不可变闭包前不会静默启用。

## 输出与质量门禁

worker 输出自包含 `EnhancedDepthBundle/1` ZIP 与 `DepthEnhancementReport/1`。报告至少包含
有效深度覆盖率、留出 RMSE/AbsRel、原始观测保持误差、深度范围和全部验收项。

已观测稀疏深度是权威测量值：worker 记录模型原始偏差后，将这些像素投影回测量值。
预测中出现 NaN、非正值、越界深度、覆盖不足或留出误差超阈值时，整个操作失败且
不发布产物。

## 安装证明与探针

安装清单 `physrealx.omnidc-installation/1` 必须记录官方源码根目录、40 位提交号、
完整 Python 源码树摘要、关键入口摘要、v1.0 checkpoint 路径与摘要，以及带摘要的
64×48 探针 RGB/稀疏深度。安装器依据该清单生成不可变部署证明。

`--probe` 会校验源码文件、revision、checkpoint SHA-256、NumPy/CUDA 环境，并实际
运行固定 RGB-D 小样本；缺少任意条件时返回 `ready=false`。
