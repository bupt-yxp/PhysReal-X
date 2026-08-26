# 资产与数字孪生

资产描述“世界中有什么以及它如何运动”。一个视觉上逼真的模型如果缺少碰撞体、关节、惯量或坐标语义，仍然不是可训练资产。

## 资产最小组成

| 组成 | 必须表达的内容 |
| --- | --- |
| 标识 | 资源 ID、语义版本、显示名称和类别 |
| 坐标 | 单位、原点、up axis、handedness 和逻辑 frame |
| 结构 | link、父子关系、可动部件和语义标签 |
| 视觉 | Mesh、USD、Gaussian、纹理、材质和 LOD |
| 碰撞 | primitive、凸分解、低模 Mesh 或 SDF |
| 运动学 | joint type、axis、limits、初始状态、damping 和 drive |
| 动力学 | mass、center of mass、inertia、friction、restitution、stiffness |
| 来源与版本 | 原始采集、构建管线、算法版本、校准数据和摘要 |
| 质量 | 几何、穿模、碰撞、稳定性、运动范围和物理一致性报告 |

## 三种接入方式

### 导入已有资产

适合已有 USD、URDF、MJCF、Mesh 或规范化 Gaussian 资产：

```python
asset = px.assets.import_asset(
    source="assets/drawer.usd",
    importer="usd",
    profile="articulated_object",
)
```

导入器只负责格式转换。转换后仍需执行结构、碰撞和物理验证。

### 从真实采集构建数字孪生

```python
job = px.assets.build(
    source="capture://drawer-session-001",
    pipeline="articulated_digital_twin",
    config={
        "reconstruction": "pd2gs",
        "collision": "mesh_decomposition",
        "estimate_joints": True,
        "estimate_physics": True,
    },
)
asset = px.jobs.wait(job).result
```

标准流水线为：

```text
采集与标定
  → 深度补全 / 几何重建
  → 部件分解与层级恢复
  → 关节类型、轴和范围估计
  → 碰撞代理生成
  → 物理参数测量或反演
  → USD / 资产清单发布
  → 仿真重播与质量验收
```

### 直接注册结构化规格

适合程序生成资产或已有完备元数据：

```python
from physrealx import AssetSpec

asset = px.assets.register(AssetSpec.from_file("drawer.asset.yaml"))
```

## 视觉与碰撞表示必须分离

高密 Mesh 或 Gaussian 适合渲染，不适合直接承担稳定碰撞。推荐：

```text
视觉表示：USD + PBR / Gaussian
碰撞表示：凸分解 / 简化 Mesh / primitive / SDF
```

两者通过相同 link 和局部坐标变换对齐，并分别保留摘要。任何重新生成碰撞体的操作都产生新资产版本。

## 铰接物体建模

每个非根 link 必须由唯一父关节连接；关节图不能有环。旋转、平移和固定关节分别声明：

- `revolute`：单位轴、角度范围、阻尼、摩擦和驱动；
- `prismatic`：单位轴、位移范围、阻尼、摩擦和驱动；
- `fixed`：上下限均为零。

抽屉示例：

```yaml
joints:
  - name: drawer.slide
    type: prismatic
    parent: cabinet
    child: drawer
    axis: [1.0, 0.0, 0.0]
    limits: [0.0, 0.4]
    damping: 2.0
    friction: 0.3
```

## 物理参数不是裸数值

每个参数应同时记录来源、不确定度和置信度：

```yaml
mass_kg:
  value: 2.0
  source: measured
  uncertainty: 0.05
  confidence: 0.98
```

来源建议限定为 `measured`、`inferred` 和 `default`。随机化范围应基于测量不确定度或明确先验，不应凭经验任意扩大。

## 校准

物理校准使用真实交互轨迹最小化仿真与真实的可观测差异：

```python
calibration = px.assets.calibrate(
    asset=asset,
    observations="dataset://drawer-real-calibration@1.0.0",
    parameters=("mass", "friction", "joint.damping"),
    objective=("joint_trajectory", "contact_force"),
)
calibrated_asset = px.jobs.wait(calibration).result
```

训练集与校准集必须分离；校准结果保存搜索范围、目标函数、随机种子、最优值和残差。

## 质量检查

```python
report = px.assets.validate(asset, profile="production")
report.raise_for_errors()
```

生产验收至少包括：

- 外部制品摘要与媒体类型正确；
- link/joint 图完整、无环且限位合法；
- 视觉与碰撞对齐，碰撞复杂度受控；
- 惯量矩阵对称正定，质量和摩擦在合理范围；
- 静态放置稳定，无初始穿模和爆炸；
- 关节在全范围内可执行，接触响应无异常；
- 真实重播误差在任务定义的容差内。

## 版本规则

- 修正文档或不影响行为的元数据：补丁版本；
- 修改碰撞、物性、关节或视觉制品：次版本；
- 改变坐标、link 命名或公共语义：主版本。

下游场景与任务引用精确摘要，因此新版本不会悄悄改变历史实验。
