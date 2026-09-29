# 物理仿真与多模态数据生成

提供 MuJoCo、Isaac Sim 4.5 物理后端，以及 Reference 契约回归后端。两个物理后端共用控制、相机、随机化和 Episode 契约，通过 `backend="mujoco"` 或 `backend="isaac_sim"` 选择。

<!-- demo-preview:simulation_and_data_generation -->

## 1. 声明任务与后端

```python
from physrealx import MethodCapability, RunSpec

backend = px.methods.bind(
    MethodCapability.SIMULATION_RUN,
    "mujoco",
    config={
        "controlled_joints": ["cabinet/drawer_slide"],
        "initial_positions": [0.0],
        "effort_limits": [20.0],
        "timestep_seconds": 0.002,
        "substeps": 25,
    },
)
policy = px.methods.bind(
    MethodCapability.POLICY_INFERENCE,
    "joint_target",
    config={"target": [0.3]},
)
run = px.runs.submit(
    RunSpec(
        scene=scene_ref,
        task=task_ref,
        backend=backend,
        policy=policy,
        episode_count=8,
        base_seed=2026,
        dataset_id="cabinet.simulation",
        dataset_version="1.0.0",
        record_signals=("joint.position", "joint.velocity"),
    )
)
dataset = px.datasets.get(px.runs.wait(run).dataset)
```

`controlled_joints` 使用 `场景实体名/关节名`，与动作向量顺序一致。位置、增量和作用力控制分别为 `joint_position`、`joint_delta`、`joint_effort`。位置控制通过有明确增益和作用力上限的 PD 控制器进入物理引擎，不直接改写关节位置。

当前固定每个资产的根节点，支持树状固定、移动和转动关节，以及 primitive box / 自包含 Mesh。碰撞 Mesh 必须已分解为凸部件。同一个动作向量使用一致的物理单位；混合移动与转动单位的向量会被拒绝。场景放置、资产根变换、link 几何变换、惯量、关节轴和限位均显式编译。

## 2. 观测契约

| 信号 | 形状、单位与 frame |
| --- | --- |
| `joint.position` | `(N,)`，`m` / `rad`，`joint` |
| `joint.velocity` | `(N,)`，`m/s` / `rad/s`，`joint` |
| `joint.effort` | `(N,)`，`N` / `N*m`，`joint` |
| `contact.force` | `(1,)`，接触强度，`N`（后端分量定义见 sensor metadata），`joint` |
| `joint.<entity>.<joint>.position/velocity/effort` | 场景内任意活动关节，包括未驱动关节，按该关节真实单位 |
| `camera.<name>.rgb` | `(H,W,3)` uint8 |
| `camera.<name>.depth` / `mask` | `(H,W)` float32 米制深度 / int32 刚体 ID |
| `camera.<name>.events` | `(H,W,2)` int32 正/负亮度阈值越过次数，帧间累计 |
| `camera.<name>.pointcloud` | `(H,W,3)` float32 米制有序点云 |
| `camera.<name>.intrinsics` | `(3,3)` float64 针孔内参 |
| `camera.<name>.position/orientation` | `(3,)` / `(4,)` 相机到世界位姿，旋转为 xyzw |
| `body.position/orientation` | `(B,3)` / `(B,4)`，世界坐标系刚体位姿 |
| `body.external_force/external_torque` | `(B,3)`，N / N*m，世界方向轴 |
| `body.wrench_origin` | `(B,3)`，显式力矩参考点；MuJoCo 为根子树质心，Isaac 为刚体质心 |

相机配置统一为 `cameras={"front": CameraSpec, "side": CameraSpec}`。配置 pose 使用 OpenGL 相机轴（-Z 向前、+Y 向上），输出深度/点云/姿态统一为 OpenCV 光学轴（+Z 向前、+Y 向下）。最多八台相机，帧率与任务周期显式一致。

图像/深度/Mask/点云/内外参以 NPY 制品保存；位姿和外力可使用内联数组。Episode.metadata.sensor_metadata 保存 Mask ID 名称、刚体数组顺序、轴约定和力矩参考点。事件为帧积分模型，不是异步 DVS 实机数据。

无显示服务器时可在启动解释器前设置 `MUJOCO_GL=egl`。图形驱动缺失会明确失败；核心 SDK 和无图像的物理运行不要求渲染上下文。

## 3. 生命周期与数据发布

```text
校验资源/能力 → 编译模型 → reset → policy.act / physics.step → close
→ Episode → DatasetManifest 原子发布
```

每个普通步骤保存当前观测、动作和随后转移的奖励；末帧仅保存终止观测。任务成功和终止判据由平台统一检查。随机种子、控制周期、后端、策略和资产版本随 Episode 保留；`randomization` 显式控制质量/惯量、阻尼/摩擦、初态、颜色、光照、相机位置和视觉遮挡。每次 reset 先恢复标称参数，再以该 episode 的种子采样；重置不累积扰动。

越界命令、单位错误、非法惯量、非凸碰撞网格、非有限状态或求解器警告均终止运行，不发布成功数据集。无论成功、异常或用户中断，session 和渲染器都关闭。

MuJoCo 对接触使用自己的求解模型和动态滑动摩擦；不能自动等同于其他仿真器的恢复系数或静摩擦语义。真实任务必须通过独立轨迹验收。

## 4. 接触与训练

`controlled_joints` 只为选中的机械臂关节安装电机；`passive_positions` 声明未驱动关节的初态。模块四沿用模块三的开源机械臂接触任务，同时记录目标关节状态与接触力；任务主体和控制机器人是不同实体。

模块四读取模块三发布的场景、任务实例和已验收轨迹。任务实例是控制关节、初态、力矩限制和被动物体初态的唯一来源；Notebook 不重新推断或复制这些参数。专家批次使用 `joint_sequence` 回放 Panda 七关节轨迹，失败批次使用同维度的初态保持策略，所有 Episode 按任务最终谓词标注。

产生的数据进入 ACT 与 DINO-WM 训练，见[训练指南](data-management-and-training.md)。核心离线执行证据协议保留为扩展入口；事件/点云/渲染直接消费同一后端记录的状态，避免另跑一份轨迹造成错位。

固定时长训练轨迹使用 `RunSpec.stop_on_success=False`，成功后继续记录，到时域末帧重新判断成败；
后端故障和显式终止条件仍优先执行。当前 Demo 生成两条随机化专家 Episode 和两条无动作失败 Episode，每条 201 帧。扩大数据规模只需增加 `RunSpec.episode_count`，不改变数据契约。

`datasets.preview_episode(..., sensor_source="blender")` 抽取 Episode 帧，用记录的刚体位姿驱动统一 Blender PBR 场景。overview 显示执行视图；front 在相同材质、光照与时刻生成 RGB Beauty、米制 Depth 和实例/Link Mask；XYZRGB 点云由这组 RGB-D 反投影。

这是离线展示重渲染。训练 Episode 中的 RGB-D 等数组仍来自所选仿真后端，预览不会替换训练数据。`sensor_source="recorded"` 才展示 Episode 实际保存的原生传感器输出。

PartNet-Mobility 资产只生成域随机化仿真数据，不声明实拍外观对齐。仅当场景实体显式绑定模块一生成的 `registered_appearance`，并且配准来源、相机和关节状态通过校核时，才可另行调用 `registered_artgs`。

## Isaac Sim 后端

配置本机已安装的 Isaac Sim **4.5.0**，由其 `python.sh` 启动独立 Kit 进程：

```bash
export PHYSREALX_ISAAC_SIM_ROOT=/path/to/isaac-sim-4.5.0
```

平台环境需要 `physical` 依赖以生成 USD；Isaac Sim 使用自己的 Python 和原生库。
资产复用 USD 封装的网格、纹理、物性和关节坐标，根节点固定。

```python
run = px.simulations.run(
    scene_ref,
    task_ref,
    backend="isaac_sim",
    config={
        "controlled_joints": ["robot/joint_1"],
        "initial_positions": [0.0],
        "effort_limits": [40.0],
        "passive_positions": {"object/hinge": 0.0},
        "timestep_seconds": 0.002,
        "substeps": 25,
    },
    policy=policy,
    dataset_id="isaac.interaction",
)
```

支持 `joint_position`、`joint_delta`、`joint_effort`。平移使用 m/N，旋转使用 rad/N·m；
策略命令受动作范围、关节范围和力矩上限约束。PD 增益及步长需与资产惯量匹配。
`passive_positions` 仅设置初态，不为被动物体添加电机。

支持上表全部相机、关节、刚体及外力矩信号。相机使用原生 RTX Beauty、
`distance_to_image_plane` 和实例分割，保留 USD 网格、纹理和 PBR 材质；
原生分割 ID 映射为固定的 `entity/link` 标签。无效/超量程深度为 0。
点云由同帧深度与实际内参反投影；颜色取同像素 RGB；事件使用共同的帧积分模型。
渲染额外更新不会推进物理时间。

`body.external_force/external_torque` 为接触与外部施加载荷，不是关节反力，
不包含重力；方向轴为世界坐标，力矩参考点由 `body.wrench_origin` 指明。
Isaac 分别累加法向接触点与切向摩擦锚点的冲量及力矩，再除以物理子步时长；
`contact.force` 为最后子步的法向接触力模长之和，
机器人接触任务按机器人与目标实体过滤。逐刚体载荷仍包含所有接触。
`joint.effort` 记录驱动力矩，不含被动阻尼/摩擦。

两个后端均支持恒定外载荷（实体/link 名、世界轴、作用于该 link 质心）：

```python
config["external_wrenches"] = {
    "object/door": {"force_n": [0, 0, 2], "torque_nm": [0, 0.1, 0]},
}
```

两个后端共用 `randomization` 配置：质量与惯量同比缩放、阻尼、摩擦、初态、
颜色、光照、材质、相机位置与视觉遮挡。每次 reset 从标称值按种子重建，避免扰动累积；
Isaac 将采样后的 USD 属性与物理缩放记录在 `sensor_metadata.randomization`。
相同种子用于复现同一后端的采样；不同引擎的求解轨迹和渲染结果不要求数值相同。

`datasets.preview_episode(..., sensor_source="recorded")` 展示数据集中实际记录的原生
RGB/Depth/Mask/Event 与对应彩色点云；左侧仍用记录位姿驱动 Blender。
默认 `sensor_source="blender"` 保留 overview/front 共用材质和光照的展示方式。
Blender 展示为离线重渲染，不替代原生传感器数据，也不作为其随机化验收证据。

进程超时、退出或非法状态均中止运行，不发布成功数据集。
进程启动故障携带日志尾部。探测到安装包只代表依赖就绪，GPU 和 Kit 的可用性在实际启动时验证。

真实引擎回归测试：

```bash
PHYSREALX_TEST_ISAAC_SIM=1 MUJOCO_GL=egl pytest tests/integration/test_physical_pipeline.py -k "isaac or native_multimodal"
```
