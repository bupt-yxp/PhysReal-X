# 第三方源码与许可声明

PhysReal-X 自有代码使用 MIT 许可证。`algorithms/` 中的内嵌方法保留各自的上游许可，
不因归入同一代码仓库而改为 MIT。

## ArtGS

- ArtGS 主体：Apache-2.0，见 `algorithms/artgs/LICENSE`。
- `diff-gaussian-rasterization`：Gaussian-Splatting 自定义许可，仅允许非商业研究和评估，
  见 `algorithms/artgs/submodules/diff-gaussian-rasterization/LICENSE.md`。
- `simple-knn`：同上述 Gaussian-Splatting 自定义许可，见
  `algorithms/artgs/submodules/simple-knn/LICENSE.md`。
- GLM：MIT 或 Happy Bunny License，见
  `algorithms/artgs/submodules/diff-gaussian-rasterization/third_party/glm/copying.txt`。
- `utils/pointnet_lib`：上游 ArtGS 快照未为该目录附带独立许可文件，因此来源清单标记为
  `NOASSERTION`；商业发布前需由使用者完成单独许可审核。

由于 ArtGS 运行依赖上述受限及待审核组件，平台方法描述符使用
`LicenseRef-ArtGS-Research-Only`，不使用 Apache-2.0 冒充整体有效许可。
