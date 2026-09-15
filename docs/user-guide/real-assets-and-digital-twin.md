# 真实资产接入与数字孪生

本模块从已经采集的 RGB-D 与可选逆向建模 USD 开始，发布可追溯的观测、米制相机、深度、实拍表面及参考部件结构。相机驱动和在线采集不属于此模块；碰撞与物性由[物理资产构建](physical-assets.md)处理。

## oven4 的当前执行路径

项目中的 `examples/real_assets_and_digital_twin/demo.ipynb` 使用当前 PhysReal-X Conda 环境，从头执行以下流程：

```text
输入与环境 → 相机与米制尺度 → 数据集 Mask 与深度增强
→ 几何与运动重建 → 位姿配准与关节校核 → 重建质量评估 → 结果与导出
```

Notebook 中所有 SDK 调用、方法和参数直接可见，不需要替换 `None`、编写额外 YAML 或启动第二个环境。

功能章节与公共 API 对应；算法名称仅作为 `method` 选择与来源记录。
共享米制输入由深度阶段生成。默认只运行用于后续配准的运动重建；CPU 表面融合是可选诊断。
表格只显示关键指标，完整操作配置和质量报告由项目 `OperationManifest` 与制品保存。

`assets.evaluate_reconstruction` 以逆向建模 USD 评估真实重建，采用固定 DTArt 指标协议，与重建 `method` 无关，输出起始及预测运动后的静态组合、逐部件和整体 CD (10⁻³ m²)，以及逐关节 Axis Ang、Axis Pos (10⁻¹ m，仅旋转) 和 Part Motion。缺失、合并或未观察到的运动保留缺项，模型配准状态与独立标注分开记录；指标定义和通用多关节输入见[资产 API](../api/assets.md#reconstruction-evaluation)。

```bash
conda activate PhysReal-X
python -m pip install -e '.[physical,colmap,artgs,demo]'
python -m ipykernel install --sys-prefix --name physreal-x --display-name PhysReal-X
python -m jupyter lab examples/real_assets_and_digital_twin/demo.ipynb
```

## 输入与版本

ArtModeling 导入器识别 `real/can/start`、`real/can/end`、`real/cmp`、`real/video`、`real/recon` 与 `sim`。本例通过 `config={"states": ["can.start", "can.end"]}` 显式选择双状态重建；视频 RGB/深度不齐全，不参与此流程。`video` 保留为源数据。本例显式使用 `aruco_6x6_50` 和共享静态 `layout_id`；Marker 角点连接视角与状态，不能单独提供米制尺度。

必须显式提供 `depth_unit="native_u16_unscaled"`、`coordinate_frame="camera_unregistered"` 和 `include_candidates=True`。后者保留配套 USD 与元数据，不代表自动通过科学验收。

`assets.ingest` 将原始导入配置与文件来源一起冻结。同版本仅复用相同来源、导入器和配置；配置不同或旧记录缺少配置证据时明确报冲突，需使用新的显式版本。不会静默复用一个排除了 USD 的旧观测集。

## USD 参考模型

```python
reference_model = px.assets.import_reference(
    observations,
    usd_path="sim/usd/model_oven.usd",
    root_link="E_body_203",
    moving_links=("E_door_197",),
)
```

`assets.import_reference` 从已导入附件读取 USD，要求显式单位，检查本地层与数据资源、拓扑和部件树，生成米制/Z-up 网格与关节记录；原始 USD、纹理和本地 MDL 一并保留。USD 几何资源缺失会失败；外部 MDL 着色器依赖单独记录，不在 CPU 数据处理时加载。

oven4 的 USD 尺寸约 0.361 × 0.238 × 0.205 m，门是 0–90° 的旋转关节。项目显式选择 `E_body_203` 为根部件、`E_door_197` 为运动部件；旁路 JSON 只作为来源证据，不覆盖 USD 的部件或关节。

这些是逆向建模记录，不自动等价于有不确定度证明的实测物性。

## 米制相机与深度

`camera.recover:colmap` 在 CPU 上处理所选状态的全部图片，并要求至少成功注册 150 帧；检查注册率、稀疏轨迹及 Marker 重投影。RGB-D 准备处理全部已注册视图。Notebook 传入 `max_frames_per_stream=None` 声明全量，帧数统计与选择由项目内部处理；项目接口的默认值与安全上限保持不变，超出上限由接口报错。重建明确指定 `training_frames_per_state=100`，在训练阶段选取每状态 100 张训练图和 50 张验证图。

`camera.align_metric:usd_depth_scale` 使用已导入的观测和 USD 参考：

1. 从 Marker 支持的桌面平面估计各状态可见高度，以 USD 尺度包络与最大端点观测求一个共用的传感器深度比例；关节运动造成的状态高度变化不会被误判为深度单位变化。
2. 将稀疏点投影深度与实际传感器深度拟合，得到 COLMAP 到米制的比例。
3. 按整视角划分拟合/检查数据，验证单状态高度稳定性、共用深度比例及跨状态 Marker 对齐的一致性。

结果标记 `MODEL_CONDITIONED`，相机进入共同 `capture_metric` 坐标系；它不是物体相对于 USD 的完整位姿注册。独立实测三维对应仍可使用 `ransac_umeyama`，两类证据不能互相冒充。

```python
calibrated_rgbd = px.assets.prepare_rgbd(
    observations,
    metric_camera,
    stride=2,
    max_frames_per_stream=None,
)
depth_operation = px.assets.enhance_depth(
    calibrated_rgbd,
    method="rgb_guided_fill",
    config={"max_rmse_m": 0.02},
)
```

准备阶段实际解码 RGB、uint16 NPY 深度和数据集二值 Mask，核对三者来源、帧索引、分辨率及取值，并使用 COLMAP 畸变模型生成射线。0 和 uint16 饱和值保持无效；其余 Mask 内正深度投影到统一米制坐标，仅保留位于观测支撑平面至已解析物体高度包络内的样本，并保存米制证据范围。该几何判定不限制相机距离，也不改变 Mask。Mask 目录缺失、帧名不一致、非二值、空白或整帧前景都会立即失败；平台不会用固定深度阈值或模型推理替代缺失标注。

`CalibratedRGBDBundle.validation_scope` 显式区分：

| 范围 | 检查与发布语义 |
| --- | --- |
| `independent_capture` | 每帧必须有另一采集的深度参考；通过后为 `PASSED` |
| `masked_observed_pixels` | 对连续表面的有效像素先遮挡再预测；结果为 `SELF_SUPERVISED`，报告剩余无效深度 |

小孔补全保留原始有效值。无局部支持或跨断层的区域不会被强制填满；自监督模式不能满足要求独立参考的生命周期门禁。

## 几何与关节

重建通过 `assets.reconstruct_geometry(geometry_input, method=..., config=...)` 执行，支持 ArtGS 与 DTArt。所选实现必须提供统一的部件/关节预测与训练、验证视图划分；更换实现时配套其参数，公共评估和展示不变。纯表面重建不能作为运动证据。

本例为 ArtGS 指定 `reference_joint="e_door_197.joint"`。项目准备层以训练视图的对象 Mask 和米制深度点云直接拟合 USD 表面，联合求解一个实拍到参考坐标系的刚体变换和两个状态的关节位置；验证视图只负责距离门禁。状态间运动被转换为 ArtGS 的对称半程初值，最终关节和几何仍由三阶段 RGB-D 训练输出。因此这是明确记录的参考引导重建，不是独立于 USD 的预测。

重建后的 `articulated_surface_icp` 只把已验证配准绑定到外观和注册相机，不再重复拟合。未启用参考引导的重建仍可在该步骤执行一次训练后配准。参考尺寸、支撑平面、连通前景和表面范围共同排除桌面、Marker 与旁物。

完整部件几何和可动结构取自 USD。`evaluate_reference_kinematics` 计算闭合/展开构型并检查关节树与限位。
两种方法均通过统一预测输出接入固定关节/CD 评估，Notebook 不依赖各自重建报告的诊断字段。方法实现与环境准备见 [ArtGS](../api/artgs.md) 和 [DTArt](../api/dtart.md)。

### 可选表面诊断

完成 Notebook 后，如需检查深度表面融合，可使用同一 `geometry_input`：

```python
diagnostic = px.assets.reconstruct_geometry(geometry_input, method="rgbd_fusion", config={})
```

此操作独立发布诊断制品，不进入默认完成判据，也不作为运动重建的前置步骤。

`geometry.reconstruct:rgbd_fusion` 消费 `MetricRGBDInput`：米制相机、增强深度及 USD 参考的 `ArtifactRef`。它在 CPU 上投影实际深度、过滤跨断层三角形，按状态输出彩色 PLY。固定每三个有效视角选一个用于验证，其余视角用于网格生成，报告验证视图表面距离中位数和 P90。不可见区域和自动部件分割没有被虚构补齐。

## 执行验收

Notebook 直接调用 `px.assets.*`，通过 `operation.output(name)` 取回具名制品，并在最后导出 `output/results.json` 供下一模块使用。外观动画通过 `preview_reconstruction_appearance` 展示重建包发布的全部状态帧，几何动画通过 `preview_reconstruction` 展示全部重建部件及 USD GT；原始模型、网格和诊断保留在 `reconstruction.zip`。此索引仅包含业务引用，不承担运行状态管理。

`examples.run_all` 在独立内核中执行 Notebook，另存 `output/execution.json` 与执行副本，并汇总到 `examples/output/notebook-acceptance.json`。每次清理所请求模块的旧结果索引；失败、跳过、超时、执行中修改源码均不能记为成功。手动运行保留最后一次成功导出，修改输入后应按顺序全部运行。

单独执行当前模块：

```python
from pathlib import Path
from examples.run_all import execute_notebooks

execute_notebooks(Path.cwd(), ("real_assets_and_digital_twin",), timeout=14500)
```

该执行器生成 `output/demo.executed.ipynb`。当前实际执行范围见[验收说明](../acceptance.md)。六模块测试仍包含 ArtGS 专属高斯渲染，切换为 DTArt 后需要单独验证后续渲染链。本模块默认导出参考 USD、米制相机、已绑定 Mask 的 RGB-D、增强深度、`reconstruction.zip`、`reconstruction-report.json`、配准结果及关节构型；可选表面诊断单独发布。

软件执行完成、模型来源可信与独立实测验收分别记录。当前数据不包含足以完成独立物性/真机验证的完整证据；历史 ArtGS 训练或本次自监督检查不会替代这些证据。
