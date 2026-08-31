# ArtGS 视觉几何重建

`geometry.reconstruct:artgs` 是 PhysReal-X 首个完整内嵌算法方法，用于从同一对象坐标系下的
多状态 RGB-D 恢复分部 Gaussian、视觉 Mesh 和形变权重。它只负责视觉几何，不把外观指标解释为
关节、碰撞或物性结论。

## 源码与环境

源码位于 `algorithms/artgs/`，包含上游 ArtGS、Gaussian rasterizer、Simple-KNN、PointNet2 与
GLM 的固定快照；`SOURCE.toml` 记录来源、提交、许可证和平台修改。所有 Python 依赖安装到当前
PhysReal-X Conda 环境：

```bash
conda activate PhysReal-X
python -m pip install --no-build-isolation -e '.[artgs]'
```

平台不读取外部 ArtGS clone。首次执行时使用当前 `sys.executable` 与本环境的 CUDA 12.1 工具链
编译仓库内三个原生扩展；构建失败、GPU 不可见或版本不匹配时，方法保持不可用。
`--no-build-isolation` 使扩展构建复用当前环境的 Torch，不会安装或启动另一个 Python 环境。

ArtGS 主体为 Apache-2.0，但必需的 Gaussian rasterizer 与 Simple-KNN 限于
非商业研究/评估；方法有效许可标识为
`LicenseRef-ArtGS-Research-Only`。统一安装不会改变上游许可边界，
商业使用前必须完成授权与 PointNet2 来源审核。

## 输入契约

`package_artgs_input(...)` 生成 `MetricMultiStateRGBDBundle/1`，必须满足：

- `CameraRigCalibration/1` 已解析为米制对象坐标；
- `EnhancedDepthBundle/1` 状态为 `PASSED`，每个 NPY 是有限正值 float32 米制深度；
- `start` 与 `end` 具有等量、多视角、刚体相机位姿和 8-bit RGBA PNG；
- `depth_frame_ids` 对训练及留出视图一一映射，禁止复用或遗漏；
- 默认同时存在独立 `transforms_test_start/end.json`。

平台从增强深度制品读取 NPY，校验 Schema、形状、单位和声明大小后确定性生成 16-bit 毫米
PNG；目录中预存的深度图不会被采信。输入 ZIP 保存原始证据引用、每帧转换关系和固定顺序。

## 执行

```python
from physrealx import PhysRealX, package_artgs_input

px = PhysRealX.open(".physrealx")
source = package_artgs_input(
    prepared_source,
    scene_id="cabinet2",
    camera_rig=metric_camera,
    enhanced_depth=enhanced_depth,
    depth_frame_ids=frame_bindings,
    num_slots=3,
    artifacts=px.artifacts,
)
operation = px.assets.reconstruct_geometry(
    source,
    method="artgs",
    config={
        "coarse_iterations": 10_000,
        "joint_iterations": 5_000,
        "reconstruction_iterations": 20_000,
        "seed": 0,
    },
)
```

同一方法、输入、配置和输出契约生成同一操作身份。执行顺序固定为粗分解、部件/关节类型预测、
联合重建与留出渲染；任一步失败都不发布 `OperationManifest`。

## 输出与门禁

| 输出 | 内容 |
| --- | --- |
| `VisualGeometryBundle/1` | 每 link PLY、Gaussian 点云、形变权重与关节信息，ZIP 成员逐项登记。 |
| `GeometryReconstructionReport/1` | 方法 revision、米制证据、训练/留出指标、Mesh 引用和质量门禁。 |

默认只有独立留出视图存在时状态才为 `PASSED`。对缺少留出视图的历史数据显式设置
`allow_training_view_evaluation=True` 后，结果固定标为 `TRAINING_VIEWS_ONLY`，不能作为后续碰撞、
物性或可训练资产的通过证据。
