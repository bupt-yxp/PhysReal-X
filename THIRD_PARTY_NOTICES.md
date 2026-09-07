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

## COLMAP / PyCOLMAP

- `pycolmap` 是 COLMAP 的官方 Python 绑定，使用 BSD-3-Clause 许可证；它通过
  `python -m pip install -e '.[colmap]'` 作为可选依赖安装，不复制进本仓库。
- `camera.recover:colmap` 只在 PyCOLMAP 导入并通过最小功能探针后注册；核心平台未安装该
  extra 时不会发布不可执行的方法。
- `opencv-python` 使用 Apache-2.0 许可证，用于已声明 ArUco/AprilTag 家族的 ID 与
  亚像素角点检测；未声明字典时不自动猜测。

## Optional physical pipeline bindings

The `physical` extra uses unmodified official distributions: NumPy 1.26.4 (BSD-3-Clause), SciPy 1.13.1 (BSD-3-Clause), Trimesh 4.5.1 (MIT), CoACD 1.0.9 (MIT), MuJoCo 3.3.7 (Apache-2.0), and OpenUSD `usd-core` 24.11 (Apache-2.0 with its upstream modified terms). Their license files remain in the installed distributions. PhysReal-X adapters are first-party MIT code; no external cloned source tree or second interpreter is required.

## Visual policy and world-model sources

- `algorithms/act/`: official LeRobot v0.3.3 ACT network, Apache-2.0, revision `b883328e6c95681ca90a18b102e4ae5e1f91e2bf`; framework integration removed, network equations retained.
- `algorithms/dinov2/`: official DINOv2 ViT encoder and required layers, Apache-2.0, revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`. The standard ViT-S/14 weights are obtained from Meta's official distribution.
- `algorithms/dino_wm/`: official causal ViT predictor, MIT, revision `0a9492fa12044b852ae9e001cc74604b79c8bb0c`. Device-safe masks, dependency injection, and the separately recorded platform residual/decoder adaptation are documented in `SOURCE.json`.

Each directory includes the upstream license and a source/adaptation manifest. Demo configuration and execution results do not claim upstream benchmark performance.
