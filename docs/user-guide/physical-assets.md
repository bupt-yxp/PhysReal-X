# 物理可信资产构建

本模块将米制视觉网格、关节结构、逐刚体测量和逐关节辨识结果组装成可运行资产，再分别验证 USD 序列化与真实动力学。

```bash
python -m pip install -e '.[physical]'
```

阶段输入均为已经写入当前制品库的 `ArtifactRef`。所有数值必须明确单位、来源与不确定度，缺失物性不会自动补默认值。

## oven4 的显式先验路径

模块二使用 `reference_asset_assembly`：读取显式投影后的 USD 部件/关节，逐 link 构建 CoACD 碰撞体，明确给出密度、摩擦、恢复系数、阻尼、关节摩擦和作用力上限。质量先验从封闭凸碰撞代理体积之和与指定密度计算（重叠体积可能重复计入），惯量为包围盒近似，全部保持 `source=default / confidence=0 / evidence_state=prior`。

相邻部件接触由关节模型表示时，显式记录 `contact_exclusions`；MuJoCo 和 USD FilteredPairsAPI 使用同一排除关系，外部推杆碰撞保持开启。原始非流形 CAD 若需要 CoACD 修复，必须显式选择 `preprocessing="coacd"`，封闭源网格检查体积偏差；开放源表面不计算无定义的体积误差，改为检查源表面与碰撞代理外表面的双向 P95 距离。没有把修复网格或模型先验当作独立测量。

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
