# 铰接表面展开

此目录是 Mesh-Untangling 官方 CUDA 实现的固定版本及 PhysReal-X 的多姿态约束扩展；来源、许可和修改清单见 `SOURCE.toml`。原算法的自适应展开流与 IPC 恢复流分开运行，所有姿态共享同一份规范部件形状，原关节轴、原点和起止角度不变。

构建需要 CMake ≥3.25、CUDA ≥12.4、C++20 编译器及 Assimp 开发包。nlohmann/json、spdlog、FTXUI 由 CMake 按固定提交获取，也可通过其 `FETCHCONTENT_SOURCE_DIR_*` 参数指定已下载源码。GPU 架构须对应本机显卡：

```bash
cmake -S algorithms/mesh_untangling -B .physrealx/native/mesh-untangling \
  -DCMAKE_BUILD_TYPE=Release -DMU_CUDA_ARCHS=89
cmake --build .physrealx/native/mesh-untangling --parallel 6 \
  --target mesh-untangle reverse-flow physrealx-projection-check
.physrealx/native/mesh-untangling/bin/Release/physrealx-projection-check
```

平台入口为 `physrealx.integrations._native_untangling`。`pose_batch` 固定原始输入；`write_case` 创建独立检查点；`run_flow` 展开；`run_recovery` 从无交叉初值执行 IPC 恢复；`decode_candidate` 验证拓扑、固定顶点和跨姿态一致性并恢复米制坐标。命令退出成功仅表示程序结束。

派生候选通过 `refine_video_articulation(..., candidate=...)` 进入已有的独立验收：实际导出网格自交、完整原始运动范围正间隙、最大形变，以及留出 RGB/Mask/深度。任何一项失败均仅输出诊断几何。表面通过后仍须重建碰撞代理并进行动力学验收。

原生 CUDA 能量用于生成展开初值。平台随后使用 `_video_surface_recovery`，将拟合帧的 RGB、部件 Mask、深度与多帧运动直接加入表面/关节联合恢复，并对每个接受的更新检查连续间隙；审核帧只参与最终独立门控。原生 IPC 本身不包含这些视频项，不应将平台的扩展声称为作者原算法。
