# 物理仿真与多模态数据生成

当前提供 MuJoCo 真实动力学后端和 Reference 契约回归后端。MuJoCo 通过现有 `runs` 服务编译已登记的场景与任务，执行关节控制，并原子发布包含同步观测的数据集。

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
| `contact.force` | `(1,)`，全部接触力范数之和，`N`，`joint` |
| `joint.<entity>.<joint>.position/velocity/effort` | 场景内任意活动关节，包括未驱动关节，按该关节真实单位 |
| `camera.<name>.rgb` | `(H,W,3)` uint8 |
| `camera.<name>.depth` / `mask` | `(H,W)` float32 米制深度 / int32 刚体 ID |
| `camera.<name>.events` | `(H,W,2)` int32 正/负亮度阈值越过次数，帧间累计 |
| `camera.<name>.pointcloud` | `(H,W,3)` float32 米制有序点云 |
| `camera.<name>.intrinsics` | `(3,3)` float64 针孔内参 |
| `camera.<name>.position/orientation` | `(3,)` / `(4,)` 相机到世界位姿，旋转为 xyzw |
| `body.position/orientation` | `(B,3)` / `(B,4)`，世界坐标系刚体位姿 |
| `body.external_force/external_torque` | `(B,3)`，N / N*m，世界方向轴 |
| `body.wrench_origin` | `(B,3)`，力矩参考点为该刚体根子树质心 |

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

`controlled_joints` 只为选中的执行器安装电机；`passive_positions` 声明未驱动关节的初态。模块四使用显式直线推杆资产接触关闭被动下层抽屉，同时记录抽屉位移与非零接触力；任务主体和控制机器人是不同实体。

产生的数据进入 ACT 与 DINO-WM 训练，见[训练指南](data-management-and-training.md)。核心离线执行证据协议保留为扩展入口；当前事件/点云/渲染直接消费同一 MuJoCo 状态，避免另跑一份轨迹造成错位。

固定时长训练轨迹使用 `RunSpec.stop_on_success=False`，成功后继续记录，到时域末帧重新判断成败；
后端故障和显式终止条件仍优先执行。模块四每条轨迹实际步进 160 次、持续 8 秒。

`renders.generate(..., method="registered_artgs")` 从已记录 episode 读取被动关节状态，
通过资产绑定的实拍–USD 配准驱动原生高斯外观，输出带时间戳的 RGB、透明度和深度。
该视图仅包含目标物体；全场景接触、推杆与多模态由 MuJoCo 提供。
当前两状态采集只支持已观测的滑动组；相机、资产来源错绑或未观测静止组发生运动时明确拒绝。
