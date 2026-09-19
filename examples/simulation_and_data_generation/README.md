# 04 · 物理仿真与多模态数据

读取模块三发布的场景、任务实例和 Panda 轨迹，通过 `physics_backend="mujoco"` 或 `"isaac_sim"` 执行名义与失败批次，同步记录位姿、动作、关节、接触和传感观测。

同步动画以逐帧刚体状态驱动统一 Blender PBR 场景：overview 相机展示执行过程，front 相机同步输出 RGB、米制深度和实例/Link Mask，并由同一 RGB-D 生成 XYZRGB 点云。Notebook 同时输出成功/失败数据集、合并数据集和质量报告。

在 PhysReal-X Conda 环境中运行 [demo.ipynb](demo.ipynb)。输出由 `output/results.json` 索引。

Isaac Sim 使用本机 4.5 安装，启动前设置 `PHYSREALX_ISAAC_SIM_ROOT`。`sensor_source="recorded"` 可直接预览所选引擎记录的原生传感数据。
