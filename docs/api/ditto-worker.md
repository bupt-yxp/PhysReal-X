# Ditto 关节恢复 worker

Ditto worker 只接受真实交互前后的米制点云证据，用于恢复静态部件、运动部件和单关节螺旋参数。它不读取 URDF、仿真 `kinematics.json` 或人工关节标注，也不从一次交互猜测机械限位。

## 输入契约

`PairedInteractionPointCloudBundle/1` 必须包含：

- 同一右手对象坐标系中的 before/after 点云与对象掩码，单位固定为米；
- 独立标定证据、相机内外参和真实交互记录；
- 不参与网络输入的 holdout 跟踪对应点，同时含静态点和运动点；
- 官方 checkpoint 制品、SHA-256、完整 Git revision 和实验配置。

当前 adapter 固定官方仓库 revision `f69579f49bbfaadc665fee511ad960f5243e1c1a`，输入 checkpoint 只允许 `Ditto_s2m` 或 `Ditto_syn` 结构。

## 运行门禁

worker 在推理前验证官方 origin、固定 revision、无已修改跟踪文件、checkpoint 摘要、CUDA、PointNet++ 编码器与五个 ConvONets 扩展。推理后还必须通过：最小点数和交互幅度、静态 holdout 对齐、部件非退化、关节类型余量、稠密轴投票离散度、holdout 螺旋变换 RMSE 及相对 identity 的改善量。

任一证据、依赖或验收项不满足时直接失败，不会生成关节。通过时输出 `ArticulationModel/1` 和 `ArticulationRecoveryReport/1`；单对交互无法证明机械端点，因此 limits 始终明确为 `unresolved`。

## 独立环境

```bash
git clone https://github.com/UT-Austin-RPL/Ditto.git /opt/physrealx/ditto
git -C /opt/physrealx/ditto checkout f69579f49bbfaadc665fee511ad960f5243e1c1a
conda env create -f workers/requirements/ditto-runtime.yaml
conda run -n physrealx-ditto python /opt/physrealx/ditto/scripts/convonet_setup.py build_ext --inplace
```

不直接使用上游 `requirements.txt` 中与 Python 3.8 不兼容的 NumPy 1.9.5 历史锁定；`ditto-runtime.yaml` 不修改上游源码，仅将依赖固定在可执行的同一代 CUDA/PyTorch 环境。安装后仍必须通过完整 probe，不能仅根据解决器成功判定 ready。

平台协议 worker 继续使用 Python 3.10+ 独立环境，仅通过 Python 3.8 推理桥调用上述官方环境，不强行把历史 PyTorch 栈安装进平台运行时。部署清单必须显式允许并固定 `PHYSREALX_DITTO_PYTHON`、`PHYSREALX_DITTO_REPOSITORY`、`PHYSREALX_DITTO_CHECKPOINT` 和 `PHYSREALX_DITTO_CHECKPOINT_DIGEST`。`--probe` 会使用该 Python 加载 checkpoint 并执行小型 CUDA 前向夹具；只验证导入成功不算 ready。
