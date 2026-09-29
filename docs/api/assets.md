# 资产 API

资产服务复用观测集、不可变制品与方法操作；完整调用见模块一、二 Notebook。

## 采集与参考

| API | 输入与输出 |
| --- | --- |
| `assets.inspect_source(path, importer="artmodeling")` | 只读检查实拍/深度/参考附件与可导入性 |
| `assets.ingest(path, resource_id=..., version=..., importer="artmodeling", config=...)` | 冻结原始文件、导入器和配置，返回观测集 VersionedRef |
| `assets.get_observations(ref)` / `list_observations()` | 读取或列出观测集 |
| `assets.resolve_observation_source(ref)` | 从观测集解析本地采集目录，校验存在性与项目路径边界 |
| `assets.import_reference(observations, usd_path=...)` | 读取附件内 USD 单位、几何、关节、材质依赖和旁路元数据冲突 |
| `assets.prepare_rgbd(observations, metric_camera, stride=4, max_frames_per_stream=8)` | 实际解码 RGB、逐帧 uint16 NPY 深度和 Mask，绑定帧及相机射线 |

ArtModeling 原始深度使用 `depth_unit="native_u16_unscaled"`、`coordinate_frame="camera_unregistered"`；`include_candidates=True` 保留配套 USD。导入器不是在线相机驱动。

共享 Marker layout_id 可约束跨状态相机位姿，但不自动提供米制尺度；来源或配置改变必须发布新版本。

## 相机、深度与重建

| API | 实际方法 | 证据范围 |
| --- | --- | --- |
| `assets.recover_cameras(observations, method=...)` | `colmap` | 稀疏恢复及 Marker 辅助；保留未解析尺度 |
| `assets.align_cameras_metric(camera, evidence, method=...)` | `ransac_umeyama` / `usd_depth_scale` | 分别需要独立实测对应 / USD 高度模型条件 |
| `assets.enhance_depth(rgbd, method=...)` | `rgb_guided_fill` | RGB 引导小孔补全，独立参考与遮挡有效像素自监督分开记录 |
| `assets.reconstruct_geometry(source, method=...)` | `rgbd_fusion` / `artgs` / `dtart` | 可见表面、高斯或 RGB-D 神经隐式重建；部件与运动统一输出 |
| `assets.evaluate_reconstruction(source, seed=2026)` | 固定 DTArt 指标 | 重建与 USD 的部件 CD、关节方向/轴线/观测行程误差；记录缺失及合并部件 |
| `assets.preview_reconstruction(evaluation, frames=24, fps=12)` | 固定服务 | 单个 GIF 同屏展示外观、全部重建部件及 USD 参考运动 |
| `assets.articulate(pair, method=...)` | `rigid_pair_fit` | 已匹配、已分部件的米制点对拟合；独立对应验收 |

### `reconstruct_geometry` 的 `method` 参数 {#reconstruct-geometry-methods}

重建方法不是独立 API。三种实现均通过
`assets.reconstruct_geometry(source, method=..., config=...)` 调用，并输出统一的
`ReconstructionPrediction`。

| `method` | 适用范围与关键参数 | 论文 |
| --- | --- | --- |
| `"rgbd_fusion"` | CPU 可见表面基线；不补造不可见几何或部件标签 | 项目内置方法 |
| `"artgs"` | 多状态 RGB-D 高斯重建；`num_slots`、`state_streams`、`training_frames_per_state`、可选 `joint_types` / `reference_joint`；需要 CUDA 与原生扩展 | [ArtGS](https://arxiv.org/abs/2502.19459) |
| `"dtart"` | 两状态 RGB-D 神经隐式重建；`joint_types`、`state_streams`、`training_frames_per_state`、`iterations`；需要 CUDA、Kaolin 与 LoFTR 权重 | [DTArt](https://arxiv.org/abs/2404.01440) |

所有方法使用同一输入契约：

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
    method="artgs",  # "rgbd_fusion" | "artgs" | "dtart"
    config={"num_slots": 2, "state_streams": ["can.start.rgb", "can.end.rgb"]},
)
```

米制、RGB/深度来源与坐标系必须绑定。ArtGS 自动准备内部目录，所有训练阶段排除验证视角；运动组数按实拍可观测运动声明。`training_frames_per_state` 为 2–2048；每三帧保留一帧用于独立视图检查，因此 100 张训练图要求每状态至少 150 个有效视图。`joint_types` 省略时由采集数据推断；提供时其数量必须等于 `num_slots - 1`。原生扩展不可执行、视图不足或验证不通过时不会发布成功操作。

ArtModeling 输入契约要求每个状态中的 `img/<frame>.png`、`depth/<frame>.npy` 与 `mask/<frame>.png` 三者按整数帧 ID 严格配对；允许真实采集丢帧造成编号不连续。Depth 必须是二维 uint16 NPY；Mask 必须是同分辨率、非空且未覆盖整帧的 uint8 二值 PNG（背景 0、目标 255）。`assets.prepare_rgbd` 不生成或修补 Mask；它在统一米制坐标中校验 Mask 内的正深度是否落入观测支撑平面与已解析的物体高度包络，拒绝的传感器离群值写为无效深度 0，并在每帧 `view.depth_validity` 中记录计数及米制证据范围。判定不设置相机距离或固定深度上限。

DTArt 的数据转换、跨状态匹配和训练在项目内部完成；验证视图不参与 LoFTR 匹配、场景归一化或训练。其输出不是 Gaussian 模型，不能交给 `registered_artgs` 渲染。配准直接读取统一 `ReconstructionPrediction` 的部件对应，不读取重建方法的私有文件。参考位姿由采集支撑平面与训练视图表面拟合，完整搜索水平朝向；预测轴及预测关节类型不参与位姿初始化，以免错误预测阻断评估或人为降低轴向误差；类型错误由统一评估报告标记为 `joint_type_mismatch`。

重建报告中的 `metrics` 保存实现提供的诊断数据，允许不同方法的字段不同。PSNR、轮廓 IoU、深度表面距离本身不属于某个算法专有，但当前实现的诊断输出不能作为公共评估的必填输入。公共关节/CD 指标由下面的 `assets.evaluate_reconstruction` 从统一几何、关节预测与参考证据计算；它不读取方法名或诊断数据。切换重建方法不改变评估调用及其表格。

## 重建质量评估 {#reconstruction-evaluation}

```python
from physrealx import ReconstructionEvaluationInput

report_ref = px.assets.evaluate_reconstruction(
    px.artifacts.put_json(
        ReconstructionEvaluationInput(
            reconstruction_report=reconstruction_report_ref,
            reference_model=reference_model,
            registration=reference_registration,
        )
    ),
)
```

输入引用统一的重建报告 `prediction`，不读取方法专属文件、不运行训练。所有重建适配器必须发布 `ReconstructionPrediction`（公开于 `physrealx`）：`parts` 保存父子关系、关节类型、初始采集坐标系下的轴方向/原点及运动量（旋转 rad、平移 m），`states` 保存各状态的部件网格引用，`motion_pair` 指定运动参数对应的有序状态对。项目在重建操作成功发布前校验契约。评估 API 是固定服务，返回报告 `ArtifactRef`，没有 `method` 参数。评估统一采用 [DTArt 官方几何实现](https://github.com/NVlabs/DigitalTwinArt/blob/master/eval/eval_mesh.py)、[关节实现](https://github.com/NVlabs/DigitalTwinArt/blob/master/utils/articulation_utils.py)和[输出规则](https://github.com/NVlabs/DigitalTwinArt/blob/master/eval/eval_results.py)。唯一配置为 `seed`，默认 2026；每侧采样数量固定为 10000。

`ReconstructionEvaluationInput.joint_names` 可显式选择本次评估的关节及其子部件；省略时评估全部，空元组仅计算整体/静态 CD。未知或重复名称会报错。该选择由项目内部执行，完整几何始终参与整体和静态组合 CD；被选中但预测缺失的部件/关节仍显示缺项。laptop1 demo 评估 `cover.joint`，机身作为静态部件。

| 指标 | API 最终数值 / 单位 |
| --- | --- |
| CD-s / CD-m / CD-w | 双向平均平方最近点距离之和 ×1000，保存为 `cd`，单位 **10⁻³ m²**。静态组合、每个可分离部件、整体分别计算；整体不是部件 CD 平均值 |
| Axis Ang | 无向轴夹角 `axis_ang_deg`，单位 ° |
| Axis Pos | 旋转轴线最短距离 ×10，保存为 `axis_pos`，单位 **10⁻¹ m**；平移关节返回 `null`，表格显示“—” |
| Part Motion | `part_motion`：旋转为 SO(3) 旋转误差 (°)，平移为位移向量误差 (m)；单位由 `part_motion_unit` 声明 |

预测网格按共享边连接三角形，保留三角形数量不少于最大连通分量 10% 的分量；参考网格不过滤。静态组合和整体先合并再过滤，避免逐部件过滤改变阈值。评估读取网格时不焊接重合顶点，保留原始拓扑。报告保存过滤前后分量/三角形数；无效输入明确报错，不沿用上游失败返回负分数的做法。CPU 最近邻与面积加权采样实现上述定义，并使用固定随机种子；不保证与 GPU 采样逐点相同。

报告契约为 `ReconstructionEvaluationReport/2`，倍率在项目内部应用一次，Notebook 直接展示最终值。旧方法名、原始 CD 字段与非 DTArt 行程幅度差字段不再提供。两次对称 CD 取平均与一次双向 CD 数学等价，不额外除以二；也不能把平方距离误标成 mm。

所有部件和状态使用同一米制刚体配准。优先使用已发布的状态网格；声明 `motion_pair` 时才允许由起始网格和预测运动推导末态，来源逐部件记录。无 `motion_pair` 时通过输入 `state_pair` 显式选择两个状态，不静默截取。`segmented=False` 的未分部件表面仍计算整体 CD，逐部件/关节指标列出 `unsegmented` 和 `null`，不会伪造关节。缺失或合并部件保留状态和 `null`，不可作为零误差。当前是 DTArt 指标在现有重建制品上的应用，不宣称复现其完整实验输入、状态或数据集协议。

`registration` 与 `pose: ReferenceEvaluationPose` 二选一。已有单关节配准提供模型拟合的状态和部件对应；未运动关节按参考下限放置，这项假设随报告保存。对于旋转关节或多个运动部件，可传入显式 `pose`：统一旋转/平移、源/目标坐标系、预测组 ID 到 USD link 的对应、覆盖全部关节的 `start_positions`/`end_positions` (m/rad)，以及 `capture_annotations` 或 `model_fitted` 来源。该评估器支持旋转/平移及多部件，自动配准目前支持单个水平轴旋转／平移关节，不覆盖任意多关节结构。

报告 `status=COMPLETED` 只表示计算完成，不表示质量合格。尺度、配准或参考状态使用同一 USD 时属于模型参考评估；不能声称独立物理实测或直接与采用其他尺度/采样协议的论文分数比较。

`assets.preview_reconstruction(evaluation_report, frames=24, fps=12)` 在一个 GIF 中展示外观、重建和参考几何。各自运动采用对应的预测或参考关节，共用观察方向与尺度；不使用参考运动替代预测。原始网格和诊断保存在重建 ZIP。动画描述采集两状态之间的运动，不是动力学仿真。

## 物理资产

| API | 方法 | 输入要求 |
| --- | --- | --- |
| `assets.register_reconstruction_geometry(reconstruction, registration)` | 固定服务 | 仅执行输入注册坐标变换，保留模块一原始顶点数、三角面和颜色；不补面、不修改位姿 |
| `assets.predict_physics(reference, registration, geometry, images, ...)` | VLM 先验 | 固定模型版本、真实图像和几何；返回物性候选范围 |
| `assets.refine_video_geometry(evidence, config=...)` | 固定服务 | 部件 Mask、米制深度、逐帧位姿和关节位置；独立帧检查 |
| `assets.refine_articulated_collision(assembly, feasibility, config=...)` | 固定服务 | 保持视觉网格、位姿、关节轴与范围不变；源面连通域细分后，以全范围接触约束序列优化局部支撑面，并硬性验收覆盖、双向误差、连续性、厚度、体积、凸性、GLB 往返和 MuJoCo 复扫；失败不发布候选资产 |
| `assets.check_physics_feasibility(assembly, config=...)` | 固定服务 | 完整范围穿插与零外力数值检查，返回可行性报告 |
| `assets.preview_physical_asset(assembly, frames=24, fps=12)` | 固定服务 | 原始视觉、碰撞代理、局部拟合误差三栏同步动画；代理颜色只表示 link 归属，逐帧红点和数值才表示 MuJoCo 最深穿插 |
| `assets.preview_physics_video(directory, frames=24, fps=8)` | 固定服务 | 同步 RGB、深度、物体/部件 Mask 动画 |
| `assets.build_collision(mesh, method=..., config=...)` | `coacd` | 明确源单位、link、误差阈值及几何预处理策略；`max_parts` 按表面积分配给各连通表面并作为全局上限 |
| `assets.calibrate_physics(trajectories, method=...)` | `bounded_inverse_dynamics` | 有单位的时间/位置/速度/加速度/作用力，独立辨识与验证轨迹 |
| `assets.assemble(evidence, method=...)` | `reference_asset_assembly` | USD 参考、逐 link 碰撞报告、全部显式物性先验 |
| `assets.assemble(evidence, method=...)` | `measured_asset_assembly` | 独立米制、部件结构、碰撞、刚体实测与关节辨识的证据闭包 |
| `assets.package(assembly, method="openusd")` | `openusd` | 组装结果，输出自包含 USD 和逐字段数值回读 |
| `assets.validate_physics(package, evidence, method="mujoco_holdout")` | `mujoco_holdout` | 第三份独立物理轨迹与精确包清单 |

先验模式返回 `AssetSpec.evidence_state=prior`。质量、惯量、摩擦、恢复系数、驱动力限制与相邻接触排除均显式保存；不能通过自己的 USD 或仿真轨迹升级为实测。完整物性证据格式见[物理资产指南](../user-guide/physical-assets.md)。

`assets.predict_physics(..., cached_prior=artifact)` 仅在图像内容、参考/注册/几何输入、模型权重与预测实现一致时复用预测；不匹配或损坏的缓存会重新预测。Notebook 只提供已有制品，缓存判定由同一服务负责。

本地 VLM 推理按当前空闲显存为权重分配预算，并为生成过程预留空间；显存不足时允许 CPU 卸载，冷推理会变慢。保持模型精度和输入配置，实际预算与设备分配写入预测来源记录。

## 发布与复用

`assets.validate(asset, profile="simulation")` 检查可仿真性；`assets.register(asset)` 发布精确版本。`assets.get(ref)` 与 `assets.list()` 读取已登记资源。

方法适配器返回 `OperationManifest`；固定评估、修正与可视化服务返回 `ArtifactRef`。操作清单，包含输入、方法/配置、产物、指标和时间。产物通过 Schema、数组/ZIP 闭包检查后才发布。需要显式复合流程时可使用核心 `MethodRecipe`；Notebook 直接调用同一 SDK，不另建流程框架。
相机恢复和 RGB-D 的 `max_frames_per_stream` 接受整数预算或 `None`（全部视图）。省略时默认分别为 32、8；安全上限分别为 4096、512。全量模式由项目统计各流帧数，超出安全上限会在处理前报错；Notebook 无需读取内部相机结构或自行计算数量。

`assets.import_reference(..., root_link=..., moving_links=(...))` 可按原始 USD 名称显式选择运动结构，其余刚体几何并入指定静态根部件，固定假设记入报告；省略两参数时保留完整结构并严格校验。项目标识规范化为小写，原 USD 路径保留，重名冲突报错。世界连接的旋转关节仅能附着到 USD 明确固定于世界的根部件，轴与枢轴世界位置不变。

碰撞细分的 `/2` 证据仅在局部贴合、覆盖与仿真检查共同通过后提供可组装候选。`preview_physical_asset(..., rejected_refinement=report)` 可查看绑定同一源资产的失败候选，动画显式标记为诊断结果，不会将其注册为资产。
