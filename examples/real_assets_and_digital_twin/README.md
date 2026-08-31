# 真实资产接入与数字孪生

`input/ArtModeling/` 保存本机 ArtModeling 原始输入数据，该目录不提交到 Git。Notebook 在每个
API 附近展示方法、参数与待替换的证据引用。打开：

```console
python -m jupyter lab examples/real_assets_and_digital_twin/demo.ipynb
```

Notebook 逐项说明并执行“导入 → COLMAP → 米制对齐 → 深度增强 → 几何重建 →
关节恢复”。任何证据缺失都会停止在对应边界。

## 六类功能

| 功能 | Demo 执行边界 | 输出与指标 |
| --- | --- | --- |
| 数据接入与检查 | `inspect_source`、`ingest` | 文件、模态、帧、附件、单位和问题统计 |
| 相机标定与位姿恢复 | `camera.recover:colmap` | 相机内外参、稀疏结构、注册率和重投影误差 |
| 米制坐标对齐 | `camera.align_metric:metric_alignment` | Sim(3)、尺度稳定性、内点率和留出 RMSE |
| 深度增强与几何重建 | `depth.enhance:omni_dc` 与显式几何配方 | 深度误差、Mesh/Gaussian/纹理/语义、Chamfer 与 F-score |
| 位姿、部件与关节恢复 | 几何配方与 `articulation.recover:ditto` | 6DoF 位姿、部件、关节轴/原点及留出重放误差 |
| 数字孪生质量管理 | 核心契约与来源复核 | 方法、权重、版本、来源、全部制品引用与聚合指标 |

PPT 中 RDFC-GAN、DTF-Net、PD²GS 和 ArtGen 表达的是功能目标。Demo 不把未开源方法
登记为可执行能力；相同输入输出由当前工作区实际安装并通过探针的 `method` 实现。

不可变导入会复制数据到工作区制品库；首次运行前应确认输出文件系统有足够空间。
