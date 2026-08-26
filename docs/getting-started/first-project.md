# 第一个完整项目

本教程建立一个可进入版本控制、可在本地和远程平台重复执行的项目，而不是只运行一段临时 Python 脚本。

## 项目目录

```text
open-drawer/
├── physrealx.yaml
├── assets/
│   ├── drawer.asset.yaml
│   └── franka.asset.yaml
├── scenes/
│   └── drawer_table.scene.yaml
├── tasks/
│   └── open_drawer.task.yaml
├── runs/
│   ├── generate_train.run.yaml
│   └── evaluate_heldout.run.yaml
├── training/
│   └── diffusion_policy.training.yaml
├── evaluation/
│   └── drawer_heldout.evaluation.yaml
└── tests/
    └── test_contracts.py
```

配置文件进入 Git；图像、点云、权重和大数据集进入制品库，由摘要引用。

## 资产规格

```yaml
schema_version: 1
resource_id: drawer.real-001
version: 1.0.0
display_name: 真实抽屉 001
frame:
  name: asset
  up_axis: z
  handedness: right
links:
  - name: cabinet
  - name: drawer
joints:
  - name: drawer.slide
    type: prismatic
    parent: cabinet
    child: drawer
    axis: [1.0, 0.0, 0.0]
    limits: [0.0, 0.4]
representations:
  visual:
    uri: artifact://sha256/REPLACE_WITH_DIGEST
    format: usd
  collision:
    uri: artifact://sha256/REPLACE_WITH_DIGEST
    format: convex_mesh
physics:
  drawer:
    mass_kg: {value: 2.0, source: measured, uncertainty: 0.05}
    static_friction: {value: 0.35, source: measured, uncertainty: 0.03}
```

生产资产还应包含惯量、质心、动态摩擦、材质、碰撞质量报告、标定数据和原始采集血缘。

## 场景规格

```yaml
schema_version: 1
resource_id: scene.drawer-table
version: 1.0.0
entities:
  - name: robot
    asset: franka.standard@1.0.0
    pose: {translation_m: [0.0, 0.0, 0.0], rotation_xyzw: [0, 0, 0, 1]}
  - name: drawer
    asset: drawer.real-001@1.0.0
    pose: {translation_m: [0.65, 0.0, 0.4], rotation_xyzw: [0, 0, 0, 1]}
sensors:
  - name: camera.front
    type: rgbd_camera
    mount: world
    rate_hz: 30
randomization:
  lighting.intensity: {distribution: uniform, range: [800, 1400]}
  drawer.physics.dynamic_friction: {distribution: uniform, range: [0.25, 0.40]}
```

## 任务规格

```yaml
schema_version: 1
resource_id: task.open-drawer
version: 1.0.0
scene: scene.drawer-table@1.0.0
instruction: 打开抽屉至少 30 厘米
initial_conditions:
  - {signal: drawer.joint_position, operator: le, value: 0.05, unit: m}
success_conditions:
  - {signal: drawer.joint_position, operator: ge, value: 0.30, unit: m}
termination_conditions:
  - {event: unsafe_contact}
action:
  kind: end_effector_delta_pose
  frame: robot.base
  rate_hz: 20
observations:
  - camera.front.rgb
  - camera.front.depth
  - robot.joint_position
  - drawer.joint_position
  - contact.force
max_steps: 300
```

## 运行规格

```yaml
schema_version: 1
resource_id: run.open-drawer-train
version: 1.0.0
scene: scene.drawer-table@1.0.0
task: task.open-drawer@1.0.0
backend: isaac_lab
policy: motion_planner
episodes: 1000
seed: 2026
parallelism: 64
dataset:
  resource_id: dataset.open-drawer-train
  version: 1.0.0
record:
  - camera.front.rgb
  - camera.front.depth
  - robot.joint_position
  - drawer.joint_position
  - contact.force
```

## 提交前校验

```bash
physrealx validate asset assets/drawer.asset.yaml
physrealx validate scene scenes/drawer_table.scene.yaml
physrealx validate task tasks/open_drawer.task.yaml
physrealx validate run runs/generate_train.run.yaml
```

校验必须在启动昂贵仿真前发现：引用缺失、坐标系错误、单位不一致、后端能力不足、动作越界、随机化非法和数据预算不合理。

## 执行与检查

```bash
physrealx project apply .
physrealx runs submit runs/generate_train.run.yaml --wait
physrealx datasets inspect dataset.open-drawer-train@1.0.0
physrealx datasets validate dataset.open-drawer-train@1.0.0 --deep
```

深度校验会抽检或全量验证制品摘要、Episode 连续性、时间戳、信号模式、NaN/Inf、成功标签与统计异常。

## 验收清单

- 相同规格、后端版本和种子可以复现实验条件；
- 资产视觉、碰撞、关节和物性均通过校验；
- 每条 Episode 包含初始与最终观测，末帧不携带动作；
- 多模态时间戳满足声明的同步容差；
- 数据集清单精确引用全部 Episode 和上游资源；
- 训练与评测使用不同场景矩阵或种子；
- 失败样本保留原始观测、状态、动作、事件和归因结果。
