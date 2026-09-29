# 01 · 真实资产与数字孪生

**输入：** `ArtModeling/laptop1` 的 start/end RGB-D、Mask 与参考 USD。

**流程：** 相机恢复 → 米制定尺 → 深度增强 → 部件与关节重建 → 配准 → 质量评估。

<!-- demo-preview:real_assets_and_digital_twin -->

## Demo 调用

```python
reference = px.assets.import_reference(
    observations,
    usd_path="sim/usd/model_laptop.usd",
    root_link="body",
    moving_links=("cover",),
)
reconstruction = px.assets.reconstruct_geometry(
    geometry_input,
    method="artgs",
    config={
        "num_slots": 2,
        "reference_joint": "cover.joint",
        "training_frames_per_state": 100,
        "seed": 2026,
    },
)
```

COLMAP 恢复相机；`usd_depth_scale` 统一尺度；`rgb_guided_fill` 补全有局部支持的深度。ArtGS 每状态使用 100 张训练图和 50 张验证图。可按同一重建契约选择 DTArt，配套其方法参数。

参考 USD 提供尺度、结构与配准依据；真实表面由采集数据重建。数据集 Mask 必须与 RGB-D 对齐，不使用参考模型替代缺失标注。

## 输出与可视化

- 米制相机、RGB-D、增强深度及来源记录；
- `reconstruction.zip`：部件网格、外观与关节预测；
- 配准、关节构型与统一 CD/关节指标；
- 一个 GIF 同时展示实拍外观、重建和参考模型。

导出索引位于 `output/results.json`，供模块二读取。评估属于参考模型条件下的重建检查；独立实测物性在模块二处理。

[资产 API 与重建方法参数](../api/assets.md#reconstruct-geometry-methods) · [下一步：物理资产](physical-assets.md)
