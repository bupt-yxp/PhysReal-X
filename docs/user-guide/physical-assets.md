# 物理可信资产构建

本模块将米制视觉几何和关节模型转换为可供仿真与训练使用的物理资产。视觉几何、碰撞几何和物理参数必须分层保存，不能互相替代。

## 完成目标

```text
视觉几何与关节 → 碰撞代理 → 物性辨识 → 完整资产组装 → OpenUSD → 物理验证
```

资产状态按证据强度区分：`prior` 表示工程先验，`identified` 表示通过实测辨识，`validated` 表示在独立留出实验中通过验证。

## 1. 构建碰撞几何

```python
collision_operation = px.assets.build_collision(
    visual_mesh,
    method=collision_method,
    config={"source_unit": "m", "link": "drawer"},
)
```

碰撞构建方法从每个 link 的视觉 Mesh 生成近似凸碰撞体。报告必须绑定输入 Mesh 制品、单位、link
名称和输出 GLB 制品；碰撞体只用于物理解算，不应覆盖高质量视觉几何。CoACD 是候选实现，
当前发行版未注册该方法。

验收关注凸部件数量、体积偏差、表面距离、穿模和目标后端的加载稳定性。非封闭或自交 Mesh 应先修复或使阶段失败。

## 2. 辨识关节物性

```python
physics_operation = px.assets.calibrate_physics(
    system_identification_evidence_ref,
    method=physics_method,
)
```

系统辨识输入至少包含实测时间戳、`q`、`qdot`、控制量或等效力/力矩、训练—留出切分和采集批次。
候选实现可使用 MuJoCo 前向模型与有界优化恢复单关节等效惯量、阻尼和摩擦；当前发行版没有
注册该适配器，也不会猜测 link 质量。

以下情况必须返回 `NOT_IDENTIFIABLE`：激励不足、设计矩阵退化、参数触边、多初值不收敛、置信区间无界，或留出误差没有显著优于工程先验。

## 3. 组装完整资产

资产组装将以下证据合并为 `AssetAssemblyBundle/1`：

- 米制相机与坐标变换；
- 每个 link 的视觉和碰撞几何；
- 已验证的部件层级、关节类型、轴、原点和限位；
- 每个刚体的质量、质心、惯量、摩擦与恢复系数；
- 每个关节的阻尼、摩擦、刚度和来源证据；
- 材质、单位、坐标轴及全部制品闭包。

核心仓库当前不附带资产组装方法。项目必须安装并注册声明 `asset.assemble` 能力的方法，再通过
显式配方执行；单关节辨识结果不能冒充全资产物性。

## 4. 封装 OpenUSD

```python
usd_operation = px.assets.package(
    assembly_ref,
    method=package_method,
)
```

候选 OpenUSD 方法只接受尺度已解析、几何闭包完整且物性已辨识或验证的组装资源。输出包含自包含
USD、资产清单和往返校验报告；所有 prim target、引用、关节单位和物性值必须能够重新打开并与输入一致。

## 5. 物理验证

至少执行：

1. 静态放置和重力稳定性；
2. 全关节范围运动与限位检查；
3. 碰撞、穿模和接触响应检查；
4. 真实轨迹在仿真中的留出回放；
5. 质量、摩擦、阻尼与状态轨迹误差统计；
6. 目标仿真后端的导入与最小交互探针。

`px.assets.validate(asset_ref, profile="digital_twin")` 检查领域资产结构；真实物理一致性还必须由已安装验证方法产生不可变报告，渲染图不能替代定量证据。

## 输出与验收

| 产物 | 必须满足 |
| --- | --- |
| 碰撞几何 | 米制、逐 link、来源 Mesh 可追溯且误差达标 |
| 物性结果 | 参数有范围、置信区间、证据引用和留出改进 |
| 资产组装 | visual/collision/articulation/body/joint physics 闭包完整 |
| USD 资产 | 自包含、可重开、单位正确、往返检查全部通过 |
| 验证报告 | 固定协议、独立数据、失败证据和方法身份完整 |

下一步进入[场景与任务生成](scenes-and-tasks.md)，将资产放入可执行任务。
