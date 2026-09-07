# 资产 API

资产服务复用观测集、不可变制品与方法操作；完整调用见模块一、二 Notebook。

## 采集与参考

| API | 输入与输出 |
| --- | --- |
| `assets.inspect_source(path, importer="artmodeling")` | 只读检查实拍/深度/参考附件与可导入性 |
| `assets.ingest(path, resource_id=..., version=..., importer="artmodeling", config=...)` | 冻结原始文件、导入器和配置，返回观测集 VersionedRef |
| `assets.get_observations(ref)` / `list_observations()` | 读取或列出观测集 |
| `assets.import_reference(observations, usd_path=...)` | 读取附件内 USD 单位、几何、关节、材质依赖和旁路元数据冲突 |
| `assets.prepare_rgbd(observations, metric_camera, stride=4, max_frames_per_stream=8)` | 实际解码 RGB/PNG16 深度，绑定帧、相机射线及对象 Mask |

ArtModeling 原始深度使用 `depth_unit="native_u16_unscaled"`、`coordinate_frame="camera_unregistered"`；`include_candidates=True` 保留配套 USD。导入器不是在线相机驱动。

共享 Marker layout_id 可约束跨状态相机位姿，但不自动提供米制尺度；来源或配置改变必须发布新版本。

## 相机、深度与重建

| API | 实际方法 | 证据范围 |
| --- | --- | --- |
| `assets.recover_cameras(observations, method=...)` | `colmap` | 稀疏恢复及 Marker 辅助；保留未解析尺度 |
| `assets.align_cameras_metric(camera, evidence, method=...)` | `ransac_umeyama` / `usd_depth_scale` | 分别需要独立实测对应 / USD 高度模型条件 |
| `assets.enhance_depth(rgbd, method=...)` | `rgb_guided_fill` | RGB 引导小孔补全，独立参考与遮挡有效像素自监督分开记录 |
| `assets.reconstruct_geometry(source, method=...)` | `rgbd_fusion` / `artgs` | 实拍可见表面 / 原生高斯运动组与连续外观 |
| `assets.articulate(pair, method=...)` | `rigid_pair_fit` | 已匹配、已分部件的米制点对拟合；独立对应验收 |

两个几何方法使用唯一输入契约：

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
    config={"num_slots": 2, "state_streams": ["can.start.001.rgb", "can.end.001.rgb"]},
)
```

米制、RGB/深度来源与坐标系必须绑定。ArtGS 自动准备内部目录，所有训练阶段排除留出视角；运动组数按实拍可观测运动声明。详细说明见 [ArtGS](artgs.md)。

## 物理资产

| API | 方法 | 输入要求 |
| --- | --- | --- |
| `assets.build_collision(mesh, method=..., config=...)` | `coacd` | 明确源单位、link、误差阈值及几何预处理策略 |
| `assets.calibrate_physics(trajectories, method=...)` | `bounded_inverse_dynamics` | 有单位的时间/位置/速度/加速度/作用力，独立辨识与验证轨迹 |
| `assets.assemble(evidence, method=...)` | `reference_asset_assembly` | USD 参考、逐 link 碰撞报告、全部显式物性先验 |
| `assets.assemble(evidence, method=...)` | `measured_asset_assembly` | 独立米制、部件结构、碰撞、刚体实测与关节辨识的证据闭包 |
| `assets.package(assembly, method="openusd")` | `openusd` | 组装结果，输出自包含 USD 和逐字段数值回读 |
| `assets.validate_physics(package, evidence, method="mujoco_holdout")` | `mujoco_holdout` | 第三份独立物理轨迹与精确包清单 |

先验模式返回 `AssetSpec.evidence_state=prior`。质量、惯量、摩擦、恢复系数、驱动力限制与相邻接触排除均显式保存；不能通过自己的 USD 或仿真轨迹升级为实测。完整物性证据格式见[物理资产指南](../user-guide/physical-assets.md)。

## 发布与复用

`assets.validate(asset, profile="simulation")` 检查可仿真性；`assets.register(asset)` 发布精确版本。`assets.get(ref)` 与 `assets.list()` 读取已登记资源。

所有处理方法返回 `OperationManifest`，包含输入、方法/配置、产物、指标和时间。产物通过 Schema、数组/ZIP 闭包检查后才发布。需要显式复合流程时可使用核心 `MethodRecipe`；Notebook 直接调用同一 SDK，不另建流程框架。
