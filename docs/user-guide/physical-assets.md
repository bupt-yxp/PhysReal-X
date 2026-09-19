# 物理可信资产构建

本模块将米制视觉网格、关节结构、逐刚体测量和逐关节辨识结果组装成可运行资产，再分别验证 USD 序列化与真实动力学。

```bash
python -m pip install -e '.[physical]'
```

阶段输入均为已经写入当前制品库的 `ArtifactRef`。所有数值必须明确单位、来源与不确定度，缺失物性不会自动补默认值。

## 功能二：真实重建、VLM 先验与仿真检查

`examples/physical_asset_building/demo.ipynb` 使用 laptop1。流程复用现有碰撞、装配和 USD 接口，不另建资产体系：

1. `px.assets.register_reconstruction_geometry(reconstruction, registration)` 将模块一的 ArtGS 逐部件网格转换到已注册的 link 静止坐标。各部件保留共享参考原点，装配时再且仅再转换一次到关节局部坐标。参考 USD 只提供拓扑、关节与坐标注册，不再充当视觉或碰撞几何。
2. `px.assets.predict_physics(reference, registration, geometry, images, python=..., model=..., revision=...)`
   在独立进程运行固定版本的 VLM；当前示例方法为 Qwen3-VL。输出逐部件质量、材质、摩擦和恢复系数，以及逐关节阻尼、摩擦和载荷预算的候选范围。重心和惯量使用显式均匀包围盒近似，`confidence=0`。
3. `px.assets.refine_video_geometry(evidence, config=...)` 对真实重建表面进行有界法向修正，并通过未参与拟合的帧检查。不修改关节轴、原点或 `start/end` 范围。
4. 装配前从最终视觉网格重建 CoACD 碰撞体；`px.assets.check_physics_feasibility(assembly)` 扫描完整锁定范围并执行零外力回放。若穿插来自误分到可动部件的局部重建残片，`px.assets.refine_articulated_geometry(assembly, feasibility)` 同步移除错误视觉面及其碰撞凸块，再校验两者表面误差并重新验收。父子碰撞、关节轴和 `start/end` 范围保持不变；超出局部范围、面积或体积阈值时拒绝自动修正。

输出包括注册后的真实重建几何、VLM 先验、视频就绪报告、可选几何修正、碰撞体、资产装配、USD 和可行性报告，以及视频观测与资产动画。

### 视频证据

`physrealx.integrations.video_geometry.VideoGeometryEvidence` 使用不可变制品引用：

- `reference`、`registration`、`source_geometry`、`capture_id`、显式 `depth_scale_m`；
- `link_ids`：每个部件对应唯一正整数，0 表示背景、手或不可用像素；
- `frames`：唯一帧 ID、`fit/check` 分组、对齐的深度 NPY、部件标签 PNG、`(fx,fy,cx,cy)`、OpenCV `camera_from_reference` 刚体矩阵以及逐关节位置。

拟合和检查不能复用同一深度制品，每个部件至少需要各两帧有效观测。输入深度不允许 NaN/Inf，无效像素使用 0。当前支持单关节参考模型、单网格部件；不支持的情况明确返回诊断。原始 RGB-D 文件不等于该证据契约：不能猜测相机位姿、帧率或手部施力，也不把 VLM 的接触判断当成力传感器数据。

当前 laptop1 视频有 186 帧同步 RGB/Depth、整物体 Mask 和部件 Mask。相机位姿尚未提供，因此 Demo 展示观测动画但不执行表面修正。手部施力未知时，动力学参数保持 VLM 先验；具有已知作用量和独立采集的数据可继续使用下文的 `bounded_inverse_dynamics`。

### 显式先验路径

显式提供参数时，`reference_asset_assembly` 读取投影后的 USD 部件/关节，逐 link 构建 CoACD 碰撞体，并使用给定的密度、摩擦、恢复系数、阻尼、关节摩擦和作用力上限。此路径的质量先验从封闭凸碰撞代理体积之和与指定密度计算（重叠体积可能重复计入），惯量为包围盒近似，保持 `source=default / confidence=0 / evidence_state=prior`。

相邻部件接触由关节模型表示时，显式记录 `contact_exclusions`；MuJoCo 和 USD FilteredPairsAPI 使用同一排除关系，外部机器人接触保持开启。原始非流形 CAD 若需要 CoACD 修复，必须显式选择 `preprocessing="coacd"`，封闭源网格检查体积偏差；开放源表面不计算无定义的体积误差，改为检查源表面与碰撞代理外表面的双向 P95 距离。没有把修复网格或模型先验当作独立测量。

下列独立辨识/组装路径仍可使用真实测量，不能由当前先验包自行提供其验收真值。

## 1. 逐 link 碰撞构建

```python
collision = px.assets.build_collision(
    visual_mesh,
    method="coacd",
    config={"source_unit": "m", "link": "e_door_197", "threshold_m": 0.005},
)
collision_report = next(o.artifact for o in collision.outputs if o.name == "collision_build_report")
```

输入为自包含 GLB、PLY 或 OBJ。严格模式要求部件封闭、绕序一致且具有正体积；显式 `preprocessing="coacd"` 允许开放源表面。已封闭且凸的部件直接保留，其他几何按米制阈值分解。输出验证凸性、适用的体积／表面误差、部件数量和 GLB 重读；外表面采样排除凸分解的内部切面。几何须已转换到相应 link 的局部坐标；算法不从图像猜测尺度。

## 2. 关节物性辨识

```python
identified = px.assets.calibrate_physics(
    identification_evidence,
    method="bounded_inverse_dynamics",
)
joint_physics = next(o.artifact for o in identified.outputs if o.name == "identified_joint_physics")
```

输入 JSON 对应 `physrealx.integrations.physics.JointIdentificationEvidence`，Schema 为 `physrealx.joint-system-identification-evidence/1`：

- `joint_name`、`joint_type`（`slide` / `hinge`）；
- `train`、`holdout`：不同采集的 NPY 制品，各列为 `time, q, qdot, qddot, force`；
- `train_capture_id`、`holdout_capture_id`：独立采集身份；
- `position_unit` 为 `m` / `rad`，`force_unit` 为 `N` / `N*m`；
- `gravity_compensated=True`：输入作用量已去除已知重力项；
- `prior_parameters`：对照先验的惯量、黏性阻尼、库仑摩擦。

方法检查时序和运动学一致性，剔除近静止区间后求解有界线性逆动力学。退化激励、触边、置信区间过宽或独立验证没有改善时拒绝发布。置信区间基于独立同方差作用量残差；该方法不处理接触耦合、未知重力、柔性或迟滞。

## 3. 完整资产组装

```python
from physrealx.integrations.assembly import AssetAssemblyInput

assembly_input = AssetAssemblyInput(
    resource_id="oven4.asset",
    version="1.0.0",
    display_name="Oven",
    camera_rig=metric_camera,
    articulation=articulation_model,
    collision_reports=tuple(per_link_collision_reports),
    body_physics=measured_body_physics,
    joint_physics=tuple(per_joint_identification_results),
)
assembled = px.assets.assemble(
    px.artifacts.put_json(assembly_input),
    method="measured_asset_assembly",
)
assembly = next(o.artifact for o in assembled.outputs if o.name == "asset_assembly")
```

`body_physics` 的 Schema 是 `physrealx.identified-body-physics/1`，包含 `status="IDENTIFIED"`、`evidence_state="identified"` 和每个 link 的 `bodies`。每项需提供：

- `link`、`evidence_artifact_id`、`center_of_mass_m`、完整 `inertia_kg_m2`；
- `mass_kg`、`static_friction`、`dynamic_friction`、`restitution`；
- 每个标量含 `value`、`unit`、`uncertainty`、`confidence`、`source`（`measured` / `inferred`），以及对应的 `evidence_artifact_id`。

组装要求全部 link 同时具有视觉/碰撞几何、全部刚体物性、全部可动关节辨识结果。惯量须正定且满足物理三角不等式。单关节等效惯量仅保留为验证证据，不替代刚体质量或惯量张量。

输出 `AssetAssemblyBundle/1` 内含统一的 `asset_spec` 和证据闭包，可用 `AssetSpec.model_validate(...)` 读取后通过 `px.assets.register(...)` 发布。资产保持 `identified`，序列化不会把它提升为 `validated`。

## 4. USD 封装与物理验收

```python
packaged = px.assets.package(assembly, method="openusd")
package_manifest = next(o.artifact for o in packaged.outputs if o.name == "asset_package_manifest")
physical_validation = px.assets.validate_physics(
    package_manifest,
    independent_validation_evidence,
    method="mujoco_holdout",
)
```

USD 输出自包含 USDZ（含纹理与已绑定高斯模型） 和逐字段重读报告。任意关节轴通过关节局部旋转表示；转动关节限位转为 USD 的角度单位。通用 USD Physics 没有库仑关节摩擦字段，该值明确保存于 `physrealx:coulombFriction` 自定义属性，其他引擎导入器须显式处理。

`PhysicalValidationEvidence` 位于 `physrealx.integrations.validation`：它绑定精确 `package_manifest`、第三份独立 `capture_id` 与 `trajectory`、单位及 `MujocoConfig`。NPY 列为时间、全部受控关节位置、速度、执行器力；采样间隔必须等于仿真控制周期。方法重读 USD 并核对完整资产，随后用该资产编译的 MuJoCo 模型回放作用量，检查全轨迹位置和速度 RMSE。

这是固定根节点、给定任务轨迹范围内的物理验证。MuJoCo 使用动态滑动摩擦及其接触求解模型；不能据此宣称全部接触材质或所有场景均与实物等价。USD 重读成功不能替代该物理报告，真实测量也不能由合成回归夹具代替。

验证脚本参见仓库 `tests/integration/test_physical_pipeline.py`。该测试真实执行凸代理、组装、USDZ、MuJoCo、数据集、训练和评测，输入明确标为合成测试数据。
