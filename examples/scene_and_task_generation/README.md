# 03 · 场景、任务与轨迹

用户设置场景类型和自然语言任务；平台从 PartNet-Mobility 选择物体，以
`physcene_guidance` 生成场景，编译并执行机械臂任务。Notebook 展示执行动画及
场景、任务脚本、操作轨迹和成功/失败样本索引。

机械臂采用 MuJoCo Menagerie 的 Franka Emika Panda，固定上游版本并随输入保留
Apache-2.0 许可证；平台导入其真实关节树、碰撞体和惯性参数。

布局采用 [PhyScene](https://github.com/PhyScene/PhyScene) 的碰撞、房间边界和可达性
guidance；当前输入为明确的 PartNet-Mobility 资产，不包含其扩散模型推理。

在 PhysReal-X Conda 环境中运行 [demo.ipynb](demo.ipynb)。输出由 `output/results.json` 索引。
