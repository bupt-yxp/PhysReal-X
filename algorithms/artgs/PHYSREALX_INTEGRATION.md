# PhysReal-X 集成边界

ArtGS 作为 `geometry.reconstruct:artgs` 内嵌方法运行。平台负责验证米制多状态 RGB-D 输入、冻结训练配置、调用三阶段训练、构建原生 CUDA 扩展、发布 Gaussian/权重/视觉 Mesh，并生成带输入摘要和质量门禁的重建报告。

源码随 PhysReal-X 主仓库维护；根项目的 `artgs` 可选依赖是唯一 Python 依赖入口。数据、预训练权重、训练输出和本机编译结果不属于源码快照。
方法执行前会核对 `SOURCE.toml` 的上游提交与源码树摘要，并拒绝从 site-packages 或工作区外部载入同名 CUDA 扩展。
平台只调用 `train_coarse.py`、`train_predict.py`、`train.py` 和 `render.py`；上游的数据下载、视频与可视化辅助脚本按原样保留供审计，不是平台可执行 API。

ArtGS 主体是 Apache-2.0，但必需的 Gaussian rasterizer 和 Simple-KNN 仅允许
非商业研究与评估，因此方法整体标记为
`LicenseRef-ArtGS-Research-Only`。详细边界见根目录
`THIRD_PARTY_NOTICES.md` 与各组件的原始许可文件。

ArtGS 的外观重建指标不能替代关节、碰撞或物性验证；它只完成数字孪生流程中的视觉几何阶段。
