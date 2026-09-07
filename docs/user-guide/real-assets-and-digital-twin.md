# 真实资产接入与数字孪生

本模块从已经采集的 RGB-D 与可选逆向建模 USD 开始，发布可追溯的观测、米制相机、深度、实拍表面及参考部件结构。相机驱动和在线采集不属于此模块；碰撞与物性由[物理资产构建](physical-assets.md)处理。

## cabinet2 的当前执行路径

项目中的 `examples/real_assets_and_digital_twin/demo.ipynb` 使用当前 PhysReal-X Conda 环境，从头执行以下流程：

```text
输入与环境 → 相机与米制尺度 → 深度准备与增强
→ 几何与运动重建 → 位姿配准与关节校核 → 结果与导出
```

Notebook 中所有 SDK 调用、方法和参数直接可见，不需要替换 `None`、编写额外 YAML 或启动第二个环境。

六个功能章节对应七个执行阶段（环境单独检查）；算法名称仅作为 `method` 选择与来源记录。
共享米制输入由深度阶段生成。默认只运行用于后续配准的运动重建；CPU 表面融合是可选诊断。
表格只显示关键指标，完整操作配置和质量报告由制品与回执保存。

```bash
conda activate PhysReal-X
python -m pip install -e '.[physical,colmap,artgs,demo]'
python -m ipykernel install --sys-prefix --name physreal-x --display-name PhysReal-X
python -m jupyter lab examples/real_assets_and_digital_twin/demo.ipynb
```

## 输入与版本

ArtModeling 导入器识别 `real/can/start`、`real/can/end`、`real/cmp`、`real/video`、`real/recon` 与 `sim`。`can.start/end` 用于双状态重建，`video` 保留为源数据。本例显式使用 `aruco_6x6_50` 和共享静态 `layout_id`；Marker 角点连接视角与状态，不能单独提供米制尺度。

必须显式提供 `depth_unit="native_u16_unscaled"`、`coordinate_frame="camera_unregistered"` 和 `include_candidates=True`。后者保留配套 USD 与元数据，不代表自动通过科学验收。

`assets.ingest` 将原始导入配置与文件来源一起冻结。同版本仅复用相同来源、导入器和配置；配置不同或旧记录缺少配置证据时明确报冲突，需使用新的显式版本。不会静默复用一个排除了 USD 的旧观测集。

## USD 参考模型

```python
reference_model = px.assets.import_reference(
    observations,
    usd_path="sim/usd/Cabinet_3.usd",
)
```

`assets.import_reference` 从已导入附件读取 USD，要求显式单位，检查本地层与数据资源、拓扑和部件树，生成米制/Z-up 网格与关节记录；原始 USD、纹理和本地 MDL 一并保留。USD 几何资源缺失会失败；外部 MDL 着色器依赖单独记录，不在 CPU 数据处理时加载。

cabinet2 的 USD 给出约 0.35 × 0.34 × 0.34 m、3 个部件与 2 个移动关节，限位约 0–0.28 m；`sim/info.json` 给出的尺寸不同。本例显式选择 USD 为模型参考，记录冲突，旁路 JSON 不覆盖 USD 的部件名称或关节坐标。

这些是逆向建模记录，不自动等价于有不确定度证明的实测物性。

## 米制相机与深度

`camera.recover:colmap` 在 CPU 上每状态均匀采样最多 32 帧，检查注册率、稀疏轨迹及 Marker 重投影。本地数据实际注册 52/64 帧，平均重投影误差约 0.886 px。

`camera.align_metric:usd_depth_scale` 使用已导入的观测和 USD 参考：

1. 从 Marker 支持的桌面平面估计可见物体高度，以 USD 高度估计传感器深度比例。
2. 将稀疏点投影深度与实际传感器深度拟合，得到 COLMAP 到米制的比例。
3. 按整视角划分拟合/检查数据，验证高度稳定性、深度比例及跨状态 Marker 对齐的一致性。

结果标记 `MODEL_CONDITIONED`，相机进入共同 `capture_metric` 坐标系；它不是物体相对于 USD 的完整位姿注册。独立实测三维对应仍可使用 `ransac_umeyama`，两类证据不能互相冒充。

```python
calibrated_rgbd = px.assets.prepare_rgbd(
    observations,
    metric_camera,
    stride=2,
    max_frames_per_stream=16,
)
depth_operation = px.assets.enhance_depth(
    calibrated_rgbd,
    method="rgb_guided_fill",
    config={"max_rmse_m": 0.02},
)
```

准备阶段实际解码 PNG16，核对 RGB 来源与帧索引，使用 COLMAP 畸变模型生成射线。0 和饱和深度保持无效。全部已恢复 Marker 在共同米制坐标中拟合桌面平面与空间范围，供每个注册视角分割物体；估计尺度所需的逐帧 Marker 可见性不限制重建视角，也不使用会裁断物体的单视图 Marker 凸包。

`CalibratedRGBDBundle.validation_scope` 显式区分：

| 范围 | 检查与发布语义 |
| --- | --- |
| `independent_capture` | 每帧必须有另一采集的深度参考；通过后为 `PASSED` |
| `masked_observed_pixels` | 对连续表面的有效像素先遮挡再预测；结果为 `SELF_SUPERVISED`，报告剩余无效深度 |

小孔补全保留原始有效值。无局部支持或跨断层的区域不会被强制填满；自监督模式不能满足要求独立参考的生命周期门禁。

## 几何与关节

默认重建通过 `assets.reconstruct_geometry(geometry_input, method=RECONSTRUCTION_METHOD, config=...)` 执行。
所选实现必须提供后续配准所需的运动组和留出评估；更换实现时也要配置其对应参数，不能把纯表面重建当作运动证据。
`articulated_surface_icp` 随后拟合实拍到 USD 的位姿和两状态关节位置，再检查留出距离。

完整部件几何和可动结构取自 USD。`evaluate_reference_kinematics` 计算闭合/展开构型并检查关节树与限位。
当前所选方法恢复两个可观测运动组，报告留出前景误差与轮廓 IoU；实现细节见[重建方法](../api/artgs.md)。

### 可选表面诊断

完成 Notebook 后，如需检查深度表面融合，可使用同一 `geometry_input`：

```python
diagnostic = px.assets.reconstruct_geometry(geometry_input, method="rgbd_fusion", config={})
```

此操作独立发布诊断制品，不进入默认完成判据，也不作为运动重建的前置步骤。

`geometry.reconstruct:rgbd_fusion` 消费 `MetricRGBDInput`：米制相机、增强深度及 USD 参考的 `ArtifactRef`。它在 CPU 上投影实际深度、过滤跨断层三角形，按状态输出彩色 PLY。固定每三个有效视角留出一个，其余视角用于网格生成，报告留出表面距离中位数和 P90。不可见区域和自动部件分割没有被虚构补齐。

## 执行验收

`output/digital-twin-report.json` 记录每阶段状态、耗时、制品和异常；失败抛出错误并保存 `FAILED`。重跑上游会使下游回执失效，全部必要阶段完成后才写入 `execution_status=COMPLETED`。

```bash
PHYSREALX_REAL_DATA_TESTS=1 python -m pytest tests/integration/test_notebook_suite.py -q
```

该测试按顺序实际运行六个 Notebook 的所有代码单元格，并生成 `output/demo.executed.ipynb`。本模块默认导出参考 USD、米制相机、增强深度、`reconstruction.zip`、`reconstruction-report.json`、配准结果及关节构型；可选表面诊断单独发布。

软件执行完成、模型来源可信与独立实测验收分别记录。当前数据不包含足以完成独立物性/真机验证的完整证据；历史 ArtGS 训练或本次自监督检查不会替代这些证据。
