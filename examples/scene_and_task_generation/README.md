# 03 · 场景、任务与轨迹

用户设置场景类型和自然语言任务；平台以 `method="SceneSmith"` 扫描完整的
PartNet-Mobility 资产库，生成多个功能匹配候选，随后布局、编译并执行机械臂任务。Notebook 展示执行动画及
场景、任务脚本、操作轨迹和成功/失败样本索引。

机械臂采用 MuJoCo Menagerie 的 Franka Emika Panda，固定上游版本并随输入保留
Apache-2.0 许可证；平台导入其真实关节树、碰撞体和惯性参数。

接入固定版本的 [SceneSmith](https://github.com/nepfaff/scenesmith) 检索与场景生成协议；
PhysReal-X 适配层将其候选转换为统一 `SceneSpec`，并交给已有的 MuJoCo/Isaac Sim
执行门控，避免在核心包中复制 SceneSmith 的 Blender、Drake 和模型运行环境。

在 PhysReal-X Conda 环境中运行 [demo.ipynb](demo.ipynb)。输出由 `output/results.json` 索引。
