---
title: 米制对齐
description: 使用独立度量证据解析 COLMAP 的 Sim(3) 尺度模糊性
---

# 米制相机与重建对齐

`metric_alignment` 是 `camera.recover` 与 `depth.enhance` 之间的独立 CPU
worker。它用已测量的 3D–3D 对应估计从 COLMAP 坐标系到米制世界坐标系的
Sim(3)，输出 `scale_status="resolved"` 的 `CameraRigCalibration/1` 和独立留出评估报告。

```python
aligned = px.assets.align_cameras_metric(
    camera_outputs["camera_rig_calibration"],
    metric_alignment_evidence,
    method="metric_alignment",
    config={
        "inlier_threshold_m": 0.005,
        "max_holdout_rmse_m": 0.01,
        "max_holdout_relative_rmse": 0.02,
    },
)
```

## 证据契约

`MetricAlignmentEvidence/1` 必须显式绑定未对齐相机制品摘要，坐标单位必须为
`m`，并提供不重叠的训练与留出 3D 对应。每组对应记录 COLMAP
`source_xyz` 与实测 `target_xyz_m`；度量基准还需记录标识、实测值和来源。

若对应由 ArUco/AprilTag 检测产生，`metric_reference` 还必须包含
`marker_family`、`marker_size_m`、`layout` 及 `layout_source_uri`。图像中检测到类似
marker 的图案只能证明“可能检测”，不能证明标记字典、尺寸或世界布局，因此不能自动猜测米制尺度。

## 验收门禁

worker 使用确定性 RANSAC 与 Umeyama 估计，并在发布前同时检查几何非退化性、
内点比例、尺度边界、留一法尺度稳定性、独立留出绝对/相对 RMSE 以及旋转行列式。
多个 COLMAP 连通分量具有独立的相似规，必须逐个对齐；worker 不会用一个尺度覆盖所有分量。
