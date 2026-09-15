---
title: DTArt 重建
description: 两状态 RGB-D 到部件几何和关节运动
---

# DTArt 重建

`geometry.reconstruct:dtart` 使用 NVIDIA DigitalTwinArt 的神经隐式重建与运动估计。
DTArt 已通过同一重建契约接入，训练视图、配准、统一指标和运动对比均由公共流程处理。当前 oven4 的默认示例选择 ArtGS；DTArt 可在同一 API 契约下替换，实际状态见[验收记录](../acceptance.md)。

```python
operation = px.assets.reconstruct_geometry(
    geometry_input,
    method="dtart",
    config={
        "joint_types": ["slide"],
        "state_streams": ["can.start.rgb", "can.end.rgb"],
        "training_frames_per_state": 100,
        "iterations": 4000,
        "seed": 2026,
    },
)
```

- 输入沿用 `MetricMultiStateRGBDBundle/1`。RGB、深度、掩膜同步去畸变；输入深度为米，DTArt PNG 中以毫米编码。
- 每状态 100 张训练图和 50 张验证图；不足直接报错。验证图不进入 LoFTR 匹配、场景归一化和训练。
- `joint_types` 指定关节输出的解释类型；上游仍优化刚体变换，当前不声称自动识别关节类型。轴、位置和状态间运动量来自重建输出。
- 训练目录不包含参考 USD 几何；当前米制尺度仍可能由 USD 辅助确定，报告保留模型条件证据范围。
- 输出使用统一 `ReconstructionPrediction`：米制部件网格、旋转弧度／平移米、公共坐标系下的关节轴。
- 验证视图表面距离是观测深度到重建表面的单向距离，不是部件 CD。部件 CD 和关节指标仍由 `assets.evaluate_reconstruction` 统一计算。
- 当前 `articulated_surface_icp` 仅支持一个观测平移运动组和参考 +X 轴；这项边界不随重建方法改变。

## 当前环境安装

在现有 PhysReal-X Conda 内执行，不运行上游创建环境脚本。Kaolin 的版本与当前 Torch 2.5.1 / CUDA 12.1 配套；其客户端约束需要配套的 ipykernel 6.29.5。

```bash
python -m pip install -e '.[dtart]' -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html
cd algorithms/dtart/mycuda
MAX_JOBS=2 python setup.py build_ext --inplace
```

LoFTR outdoor 权重来自 [Kornia 官方仓库](https://huggingface.co/kornia/loftr)，放置在 `algorithms/dtart/weights/outdoor_ds.ckpt`。权重不纳入 Git。
SHA256：`21f5bec5968178e8bc8b7633441836fe5de4f47d861dd2cd7dc38e271b0479ec`。
`px.methods.inspect(MethodCapability.GEOMETRY_RECONSTRUCT, "dtart")` 检查运行条件。原生导出包包含模型权重、双状态网格、运动参数和训练日志。

DTArt 输出不是 Gaussian 模型，不能交给 `registered_artgs` 渲染。选择 DTArt 时，依赖 Gaussian 外观的后续模块必须使用兼容渲染方法后再单独验收。
